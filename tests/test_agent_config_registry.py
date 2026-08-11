import sys
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agent  # noqa: E402,F401 - importing agent populates the runtime registry
from common.registry import Registry  # noqa: E402
from scripts.evaluate_hyperlight import build_arg_parser  # noqa: E402


class AgentConfigRegistryTest(unittest.TestCase):
    def test_every_runnable_config_stem_is_registered(self):
        config_stems = {
            path.stem for path in (PROJECT_ROOT / "configs" / "tsc").glob("*.yml")
        }
        config_stems.remove("base")

        registered_models = set(Registry.mapping["model_mapping"])
        self.assertEqual(set(), config_stems - registered_models)

    def test_retired_configs_are_not_exposed(self):
        for stem in ("h2tsc", "hyperlight", "maddpg", "dqn_backup"):
            with self.subTest(stem=stem):
                self.assertFalse(
                    (PROJECT_ROOT / "configs" / "tsc" / f"{stem}.yml").exists()
                )

        self.assertNotIn("h2tsc", Registry.mapping["model_mapping"])
        self.assertFalse((PROJECT_ROOT / "agent" / "h2tsc_agent.py").exists())
        self.assertFalse((PROJECT_ROOT / "trainer" / "h2tsc_trainer.py").exists())
        self.assertNotIn(
            "h2tsc_agent", (PROJECT_ROOT / "agent" / "__init__.py").read_text()
        )

    def test_hyperlight_variants_use_matching_model_names(self):
        expected_names = {
            "hyperlight_spo": "hyperlight_spo",
            "hyperlight_maspo": "hyperlight_maspo",
        }
        for stem, expected_name in expected_names.items():
            with self.subTest(stem=stem):
                path = PROJECT_ROOT / "configs" / "tsc" / f"{stem}.yml"
                config = yaml.safe_load(path.read_text())
                self.assertEqual(expected_name, config["model"]["name"])

    def test_hyperlight_evaluator_defaults_to_a_runnable_config(self):
        args = build_arg_parser().parse_args([])
        self.assertEqual("hyperlight_ppo", args.agent)
        self.assertTrue(
            (PROJECT_ROOT / "configs" / "tsc" / f"{args.agent}.yml").exists()
        )


if __name__ == "__main__":
    unittest.main()
