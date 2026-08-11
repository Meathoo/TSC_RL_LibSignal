import csv
import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import avg_compare
import compare


def avg_record(value: float, episode: int = 0, metric: str = "travel_time") -> avg_compare.Record:
    values = {
        "travel_time": 10.0,
        "loss": 1.0,
        "reward": 2.0,
        "queue": 3.0,
        "delay": 4.0,
        "throughput": 5.0,
    }
    values[metric] = value
    return avg_compare.Record(
        algo="algo",
        mode="TEST",
        episode=episode,
        **values,
    )


def loaded_seed(seed_id: str, value: float, group: str) -> avg_compare.LoadedSeed:
    return avg_compare.LoadedSeed(
        group_key=group,
        group_label=group,
        seed_label=f"seed{seed_id}",
        log_path=Path(f"/{group}/seed{seed_id}/run_DTL.log"),
        records=[avg_record(value)],
        seed_id=seed_id,
    )


class MovingAverageTests(unittest.TestCase):
    def test_even_window_never_expands_to_window_plus_one(self) -> None:
        values = [float(value) for value in range(11)]
        expected = [sum(values[:10]) / 10] * 5 + [sum(values[1:]) / 10] * 6
        self.assertEqual(avg_compare.moving_average(values, 10), expected)
        self.assertEqual(compare.apply_moving_average(values, 10), expected)

    def test_series_equal_to_window_is_smoothed(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        expected = [2.5] * 4
        self.assertEqual(avg_compare.moving_average(values, 4), expected)
        self.assertEqual(compare.apply_moving_average(values, 4), expected)


class SeedStatisticsTests(unittest.TestCase):
    def test_sample_std_is_undefined_for_one_seed(self) -> None:
        self.assertTrue(math.isnan(avg_compare.std_value([3.0], ddof=1)))
        self.assertEqual(avg_compare.std_value([3.0], ddof=0), 0.0)

    def test_seed_id_is_parsed_from_nearest_path_component(self) -> None:
        path = Path("/runs/model_seed9_variant/seed02/logger/run_DTL.log")
        self.assertEqual(avg_compare.parse_seed_id(path), "2")

    def test_pairwise_means_and_wins_use_only_matching_seed_ids(self) -> None:
        baseline = [
            loaded_seed("0", 100.0, "baseline"),
            loaded_seed("1", 1000.0, "baseline"),
        ]
        candidate = [loaded_seed("1", 900.0, "candidate")]

        rows = avg_compare._build_pairwise_comparison_rows(
            baseline_seeds=baseline,
            candidate_seeds=candidate,
            modes=["TEST"],
            metrics=["travel_time"],
            comparison_statistics=["last"],
            episode_start=None,
            episode_end=None,
            table_ma_window=1,
            decimals=2,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["baseline"], "1000.00")
        self.assertEqual(rows[0]["candidate"], "900.00")
        self.assertEqual(rows[0]["improve_%"], "10.00")
        self.assertEqual(rows[0]["win_rate"], "1/1")
        self.assertEqual(rows[0]["seeds"], "1")

    def test_pairwise_comparison_does_not_fall_back_to_position(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            rows = avg_compare._build_pairwise_comparison_rows(
                baseline_seeds=[loaded_seed("0", 100.0, "baseline")],
                candidate_seeds=[loaded_seed("1", 90.0, "candidate")],
                modes=["TEST"],
                metrics=["travel_time"],
                comparison_statistics=["last"],
                episode_start=None,
                episode_end=None,
                table_ma_window=1,
                decimals=2,
            )
        self.assertEqual(rows, [])
        self.assertIn("no unambiguous matching seed IDs", output.getvalue())


class InputIntegrityTests(unittest.TestCase):
    def test_nonfinite_metric_is_reported_and_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "seed0_DTL.log"
            log_path.write_text(
                "algo\tTEST\t0\t10\tnan\t2\t3\t4\t5\n"
                "algo\tTEST\t1\t11\t6\t2\t3\t4\t5\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                records = avg_compare.load_log(log_path)

        self.assertIn("non-finite loss", output.getvalue())
        self.assertEqual(
            avg_compare.seed_metric_values(records, "TEST", "loss", None, None, 1),
            [6.0],
        )

    def test_compare_series_ignores_nonfinite_metric_without_dropping_other_metrics(self) -> None:
        records = [
            compare.Record("algo", "TEST", 0, 10, math.nan, 2, 3, 4, 5),
            compare.Record("algo", "TEST", 1, 11, 6, 2, 3, 4, 5),
        ]
        episodes, losses = compare.records_to_series(records, "TEST", "loss")
        travel_episodes, travel_times = compare.records_to_series(
            records, "TEST", "travel_time"
        )
        self.assertEqual((episodes, losses), ([1], [6.0]))
        self.assertEqual((travel_episodes, travel_times), ([0, 1], [10.0, 11.0]))

    def test_brf_lookup_requires_exact_dtl_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            dtl_path = directory / "run_DTL.log"
            dtl_path.touch()
            (directory / "other_BRF.log").write_text(
                "Final Travel Time is 999, mean rewards: 1, queue: 2, delay: 3, throughput: 4\n",
                encoding="utf-8",
            )
            self.assertEqual(avg_compare.candidate_brf_logs(dtl_path), [])
            self.assertIsNone(avg_compare.load_final_checkpoint_record(dtl_path))

            exact_path = directory / "run_BRF.log"
            exact_path.write_text(
                "Final Travel Time is 12, mean rewards: 1, queue: 2, delay: 3, throughput: 4\n",
                encoding="utf-8",
            )
            self.assertEqual(avg_compare.candidate_brf_logs(dtl_path), [exact_path])
            self.assertEqual(
                avg_compare.load_final_checkpoint_record(dtl_path).travel_time,
                12.0,
            )


class CsvConsistencyTests(unittest.TestCase):
    def test_compare_summary_uses_plot_filter_and_smoothing(self) -> None:
        original_range = compare.EPISODE_RANGE
        original_use_ma = compare.USE_MOVING_AVERAGE
        original_window = compare.MOVING_AVERAGE_WINDOW
        compare.EPISODE_RANGE = (1, 2)
        compare.USE_MOVING_AVERAGE = True
        compare.MOVING_AVERAGE_WINDOW = 2
        records = [
            compare.Record("algo", "TEST", 0, 0, 0, 0, 0, 0, 0),
            compare.Record("algo", "TEST", 1, 10, 0, 0, 0, 0, 0),
            compare.Record("algo", "TEST", 2, 20, 0, 0, 0, 0, 0),
            compare.Record("algo", "TEST", 3, 100, 0, 0, 0, 0, 0),
        ]
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                compare.build_summary_csv(
                    [("algo", "unused.log", records)], "TEST", tmpdir
                )
                with (Path(tmpdir) / "summary_TEST.csv").open(
                    newline="", encoding="utf-8"
                ) as file:
                    rows = list(csv.DictReader(file))
        finally:
            compare.EPISODE_RANGE = original_range
            compare.USE_MOVING_AVERAGE = original_use_ma
            compare.MOVING_AVERAGE_WINDOW = original_window

        travel_time = next(row for row in rows if row["metric"] == "travel_time")
        self.assertEqual(travel_time["count"], "2")
        self.assertEqual(float(travel_time["first"]), 15.0)
        self.assertEqual(float(travel_time["last"]), 15.0)
        self.assertEqual(travel_time["episode_start"], "1")
        self.assertEqual(travel_time["episode_end"], "2")
        self.assertEqual(travel_time["moving_average_window"], "2")

    def test_paper_table_marks_insufficient_seed_count(self) -> None:
        loaded = {"candidate": [loaded_seed("0", 10.0, "candidate")]}
        with tempfile.TemporaryDirectory() as tmpdir:
            output = io.StringIO()
            with redirect_stdout(output):
                avg_compare.write_paper_table_csv(
                    loaded=loaded,
                    modes=["TEST"],
                    metrics=["travel_time"],
                    output_dir=Path(tmpdir),
                    episode_start=None,
                    episode_end=None,
                    statistics=["last"],
                    ddof=1,
                    decimals=2,
                    table_ma_window=1,
                )
            with (Path(tmpdir) / "paper_table_long_TEST.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                row = next(csv.DictReader(file))

        self.assertIn("at least 2 required", output.getvalue())
        self.assertEqual(row["mean"], "10.00")
        self.assertEqual(row["std"], "")
        self.assertEqual(row["mean_std"], "")
        self.assertEqual(row["status"], "insufficient_seeds:1<2")


if __name__ == "__main__":
    unittest.main()
