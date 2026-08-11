#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


METRICS = ["travel_time", "loss", "reward", "queue", "delay", "throughput"]
MODES = ["TRAIN", "TEST"]
TABLE_STATISTICS = ["last", "best", "mean", "tail5", "best_checkpoint", "final_checkpoint"]
FINAL_RESULT_RE = re.compile(
    r"Final Travel Time is\s+([-+0-9.eE]+),\s+"
    r"mean rewards:\s+([-+0-9.eE]+),\s+"
    r"queue:\s+([-+0-9.eE]+),\s+"
    r"delay:\s+([-+0-9.eE]+),\s+"
    r"throughput:\s+([-+0-9.eE]+)"
)
SEED_ID_RE = re.compile(r"(?:^|[^A-Za-z0-9])seed[_-]?(\d+)(?!\d)", re.IGNORECASE)


@dataclass(frozen=True)
class SeedGroup:
    key: str
    label: str
    paths: List[str]


@dataclass
class Record:
    algo: str
    mode: str
    episode: int
    travel_time: float
    loss: float
    reward: float
    queue: float
    delay: float
    throughput: float


@dataclass
class LoadedSeed:
    group_key: str
    group_label: str
    seed_label: str
    log_path: Path
    records: List[Record]
    final_record: Optional[Record] = None
    seed_id: Optional[str] = None


@dataclass
class AvgSeries:
    group_key: str
    group_label: str
    mode: str
    metric: str
    episodes: List[int]
    mean: List[float]
    std: List[float]
    min_value: List[float]
    max_value: List[float]
    seed_count: List[int]


# Default groups mirror the two methods currently enabled in compare.py.
SEED_GROUPS: List[SeedGroup] = [
    # SeedGroup(
    #     key="mat_queue",
    #     label="mat+queue",
    #     paths=[
    #         # 4x4
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mat/cityflow4x4/mat_seed0/logger/2026_06_15-18_56_33_DTL.log",

    #     ],
    # ),
    # SeedGroup(
    #     key="matSmall_queue",
    #     label="matSmall+queue",
    #     paths=[
    #         # 4x4
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mat/cityflow4x4/matSmall_seed0/logger/2026_06_16-05_41_56_DTL.log",

    #     ],
    # ),
    # SeedGroup(
    #     key="matSmall_autoregressive_queue",
    #     label="matSmall_autoregressive+queue",
    #     paths=[
    #         # 4x4
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mat/cityflow4x4/matSmall_autoregressive_seed0/logger/2026_06_16-08_54_12_DTL.log",

    #     ],
    # ),
    # SeedGroup(
    #     key="dqn",
    #     label="DQN",
    #     paths=[
    #         # grid4x4
    #         # 4x4

    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_dqn/cityflow7x28/seed0/logger/2026_05_11-19_32_44_DTL.log",
    #     ],
    # ),
    
    # SeedGroup(
    #     key="mappo_obs",
    #     label="MAPPO",
    #     paths=[
    #         # 4x4
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow4x4/queue+phase/seed0_queue+phase/logger/2026_07_11-14_41_54_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow4x4/queue+phase/seed1_queue+phase/logger/2026_07_11-15_51_15_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow4x4/queue+phase/seed2_queue+phase/logger/2026_07_11-17_02_09_DTL.log",
    #         # 16x3
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow16x3/queue+phase_seed0/logger/2026_07_15-21_14_01_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow16x3/queue+phase_seed1/logger/2026_07_15-23_31_54_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow16x3/queue+phase_seed2/logger/2026_07_16-01_49_29_DTL.log",
    #         # 7x28
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow7x28/queue+phase/seed0_queue+phase/logger/2026_07_08-18_09_46_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow7x28/queue+phase/seed1_queue+phase/logger/2026_07_09-04_00_36_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow7x28/queue+phase/seed2_queue+phase/logger/2026_07_09-18_02_34_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="mappo_obs+One-Hot(id)",
    #     label="MAPPO+One-Hot(id)",
    #     paths=[
    #         # 4x4
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow4x4/onehot/seed0_onehot/logger/2026_07_11-11_18_39_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow4x4/onehot/seed1_onehot/logger/2026_07_11-12_25_57_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow4x4/onehot/seed2_onehot/logger/2026_07_11-13_33_58_DTL.log",
    #         # 16x3
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow16x3/onehot_seed0/logger/2026_07_16-04_06_56_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow16x3/onehot_seed1/logger/2026_07_16-06_18_01_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow16x3/onehot_seed2/logger/2026_07_16-08_29_38_DTL.log",
    #         # 7x28
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow7x28/queue+phase+onehot(id)/seed0/logger/2026_06_21-11_03_35_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow7x28/queue+phase+onehot(id)/seed1/logger/2026_06_20-01_38_51_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow7x28/queue+phase+onehot(id)/seed2/logger/2026_06_22-04_33_18_DTL.log",
    #     ],
    # ),
    SeedGroup(
        key="mappo_obs+Learned(id)",
        label="MAPPO+Learned(id)",
        paths=[
            # 4x4
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo_learned/cityflow4x4/seed0_learned64/logger/2026_07_11-07_57_23_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo_learned/cityflow4x4/seed1_learned64/logger/2026_07_11-09_04_21_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo_learned/cityflow4x4/seed2_learned64/logger/2026_07_11-10_11_42_DTL.log",
            # 16x3
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo_learned/cityflow16x3/learned64_seed0/logger/2026_07_16-10_40_27_DTL.log",
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo_learned/cityflow16x3/learned64_seed1/logger/2026_07_16-12_50_47_DTL.log",
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo_learned/cityflow16x3/learned64_seed2/logger/2026_07_16-15_00_25_DTL.log",
            # 7x28
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo/cityflow7x28/queue+phase+learned64(id)/seed0_learned64/logger/2026_07_10-06_15_19_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo_learned/cityflow7x28/learned64_seed1/logger/2026_07_19-04_55_24_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_native_mappo_learned/cityflow7x28/learned64_seed2/logger/2026_07_19-15_20_42_DTL.log",
        ],
    ),
    # SeedGroup(
    #     key="onehot_queue",
    #     label="HyperLightMAPPO_One-Hot(id)",
    #     paths=[
    #         # 4x4
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/seed0_onehot/logger/2026_07_12-08_44_10_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/seed1_onehot/logger/2026_07_12-09_53_11_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/seed2_onehot/logger/2026_07_12-11_02_37_DTL.log",
    #         # 16x3
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/onehot_seed0/logger/2026_07_16-17_30_07_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/onehot_seed1/logger/2026_07_16-19_44_31_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/onehot_seed2/logger/2026_07_16-21_58_01_DTL.log",

    #         # 7x28
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0_onehot_mlp_queue_ep250/logger/2026_06_12-08_58_26_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed1_onehot_mlp_queue_ep250/logger/2026_06_13-03_08_15_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed2_onehot_mlp_queue_ep250/logger/2026_06_13-21_21_34_DTL.log",

    #     ],
    # ),
    SeedGroup(
        key="learned64_queue",
        label="all_weights",#"HyperLightMAPPO_Learned(id)",
        paths=[
            # grid4x4
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow_grid4x4/grid4x4_seed0_learned64_mlp_queue_ep250/logger/2026_06_23-23_57_32_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow_grid4x4/grid4x4_seed1_learned64_mlp_queue_ep250/logger/2026_06_24-00_59_44_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow_grid4x4/grid4x4_seed2_learned64_mlp_queue_ep250/logger/2026_06_24-02_01_40_DTL.log",  

            # 4x4
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/seed0_learned64/logger/2026_07_11-18_12_14_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/seed1_learned64/logger/2026_07_11-19_21_47_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/seed2_learned64/logger/2026_07_11-20_31_18_DTL.log",

            # 16x3
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/learned64_seed0/logger/2026_07_17-00_13_10_DTL.log",
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/learned64_seed1/logger/2026_07_17-02_26_37_DTL.log",
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/learned64_seed2/logger/2026_07_17-04_42_53_DTL.log",

            # 7x28
            # "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0_learned64_mlp_queue_ep250/logger/2026_05_31-19_10_31_DTL.log",
            # "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed1_learned64_mlp_queue_ep250/logger/2026_06_05-09_31_00_DTL.log",
            # "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed2_learned64_mlp_queue_ep250/logger/2026_06_05-20_10_30_DTL.log",
        ],
    ),
    SeedGroup(
        key="FiLM",
        label="FiLM",
        paths=[
            # grid4x4
            # 4x4
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/hyperlight_film_both_mlp_seed0/logger/2026_07_23-14_21_59_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/hyperlight_film_both_mlp_seed1/logger/2026_07_23-15_33_49_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/hyperlight_film_both_mlp_seed2/logger/2026_07_23-16_44_33_DTL.log",
            #16x3
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperlight_film_both_mlp_seed0/logger/2026_07_23-17_55_33_DTL.log",
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperlight_film_both_mlp_seed1/logger/2026_07_23-20_21_36_DTL.log",
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperlight_film_both_mlp_seed2/logger/2026_07_23-22_42_34_DTL.log",
            
            # 7x28
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/hyperlight_film_both_mlp_seed0/logger/2026_07_24-01_00_41_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/hyperlight_film_both_mlp_seed1/logger/2026_07_24-11_29_23_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/hyperlight_film_both_mlp_seed2/logger/2026_07_24-22_21_44_DTL.log",
        ],
    ),
    # SeedGroup(
    #     key="hyperlight_graph_mappo",
    #     label="graph embedding",#"HyperLightMAPPO_Learned(id)",
    #     paths=[
    #         # grid4x4

    #         # 4x4

    #         # 16x3

    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_graph_mappo/cityflow7x28/seed0_graph_mappo/logger/2026_07_22-05_21_50_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_graph_mappo/cityflow7x28/seed1_graph_mappo/logger/2026_07_22-16_12_04_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_graph_mappo/cityflow7x28/seed2_graph_mappo/logger/2026_07_23-03_11_08_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="cos",
    #     label="cos",
    #     paths=[
    #         # grid4x4
            

    #         # 4x4
    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed0_cos_learned64_mlp_queue_ep250/logger/2026_06_24-08_25_35_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed1_cos_learned64_mlp_queue_ep250/logger/2026_06_24-19_56_14_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed2_cos_learned64_mlp_queue_ep250/logger/2026_06_25-06_28_38_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed0_cosTopk16_learned64_mlp_queue_ep250/logger/2026_06_25-19_22_23_DTL.log",
    #     ],
    # ),
    
    # SeedGroup(
    #     key="cosTopk24_topo64",
    #     label="cosTopk24_topo64",
    #     paths=[
    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed0_cosTopk24_topo64_learned64_mlp_queue_ep250/logger/2026_06_29-16_52_51_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed1_cosTopk24_topo64_learned64_mlp_queue_ep250/logger/2026_06_30-03_53_25_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed2_cosTopk24_topo64_learned64_mlp_queue_ep250/logger/2026_06_30-15_05_32_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="cosTopk16_add_topo64_learned64_mlp_queue_ep250",
    #     label="cosTopk16_add_topo64_learned64_mlp_queue_ep250",
    #     paths=[
    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed0_cosTopk16_add_topo64_learned64_mlp_queue_ep250/logger/2026_07_01-07_54_30_DTL.log",
    #         # "",
    #         # "",
    #     ],
    # ),
    # SeedGroup(
    #     key="topo64",
    #     label="topo64",
    #     paths=[
    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0_topo64_learned64_mlp_queue_ep250/logger/2026_06_27-08_45_44_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed1_topo64_learned64_mlp_queue_ep250/logger/2026_06_28-18_44_32_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed2_topo64_learned64_mlp_queue_ep250/logger/2026_06_29-05_02_22_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="cos_topK16_topo",
    #     label="cos_topK16_topo",
    #     paths=[
    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed0_cosTopk16_topo64_learned64_mlp_queue_ep250/logger/2026_06_26-19_31_46_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed1_cosTopk16_topo64_learned64_mlp_queue_ep250/logger/2026_06_27-19_44_26_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed2_cosTopk16_topo64_learned64_mlp_queue_ep250/logger/2026_06_28-06_34_36_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="cos_topK16",
    #     label="cos_topK16",
    #     paths=[
    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed0_cosTopk16_learned64_mlp_queue_ep250/logger/2026_06_25-19_22_23_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="cos_topK16_policyCoef05",
    #     label="cos_topK16_policyCoef05",
    #     paths=[
    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo_cos/cityflow7x28/seed0_cosTopk16_policyCoef05_learned64_mlp_queue_ep250/logger/2026_06_26-07_53_51_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="learned64_lora_r4_s002",
    #     label="learned64_lora_r4_s002",
    #     paths=[
    #         # grid4x4
    #         # 4x4

    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0_learned64_lora_r4_s002/logger/2026_07_05-07_31_04_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed1_learned64_lora_r4_s002/logger/2026_07_05-19_24_15_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed2_learned64_lora_r4_s002/logger/2026_07_06-06_28_44_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="learned64_residual002_a0025_v0015",
    #     label="learned64_residual002_a0025_v0015",
    #     paths=[
    #         # grid4x4
    #         # 4x4

    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0_learned64_residual_a0025_v0015/logger/2026_07_03-19_10_39_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed1_learned64_residual_a0025_v0015/logger/2026_07_04-06_25_52_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed2_learned64_residual_a0025_v0015/logger/2026_07_04-18_40_40_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="learned64_residual002_a003_v002",
    #     label="learned64_residual002_a003_v002",
    #     paths=[
    #         # grid4x4
    #         # 4x4

    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0_learned64_residual002_a003_v002/logger/2026_07_03-06_03_10_DTL.log",
    #     ],
    # ),
    
    # SeedGroup(
    #     key="hyperlight_graph_mappo",
    #     label="id+topology embedding",
    #     paths=[
    #         #grid4x4
    #         #4x4
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_graph_mappo/cityflow4x4/seed0_graph_mappo/logger/2026_07_12-20_32_17_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_graph_mappo/cityflow4x4/seed1_graph_mappo/logger/2026_07_12-21_48_30_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_graph_mappo/cityflow4x4/seed2_graph_mappo/logger/2026_07_12-23_04_00_DTL.log",
    #         #7x28
    #         # "",
    #         # "",
    #         # "",
    #     ],
    # ),
    # SeedGroup(
    #     key="learned128_queue",
    #     label="learnable embedding dim 128",
    #     paths=[
    #         # 4x4

    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0_learned128_mlp_queue_ep250/logger/2026_06_17-13_03_15_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed1_learned128_mlp_queue_ep250/logger/2026_06_18-05_14_05_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed2_learned128_mlp_queue_ep250/logger/2026_06_18-20_31_39_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="learned_queuepress02",
    #     label="learned+queuePress0.2/0",
    #     paths=[
    #         # 4x4
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/hz4x4_seed0_learned64_mlp_queuePress02_ep250/logger/2026_06_15-02_37_09_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/hz4x4_seed1_learned64_mlp_queuePress02_ep250/logger/2026_06_15-04_11_02_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/hz4x4_seed2_learned64_mlp_queuePress02_ep250/logger/2026_06_15-05_44_00_DTL.log",



    #         # 7x28
    #         "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0_learned64_mlp_queuePress02_ep250/logger/2026_06_02-18_53_47_DTL.log",
    #         "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed1_learned64_mlp_queuePress02_ep250/logger/2026_06_04-19_14_08_DTL.log",
    #         "data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed2_learned64_mlp_queuePress02_ep250/logger/2026_06_08-20_58_19_DTL.log",
    #     ],
    # ),
    SeedGroup(
        key="residual002",
        label="all_weights_residual",#"HyperLightMAPPO+residual002",
        paths=[
            # grid4x4
            
            # 4x4
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/seed0_res02/logger/2026_07_11-21_40_20_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/seed1_res02/logger/2026_07_11-22_55_44_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/seed2_res02/logger/2026_07_12-00_10_54_DTL.log",
            
            # 16x3
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/res002_seed0/logger/2026_07_17-06_56_49_DTL.log",
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/res002_seed1/logger/2026_07_17-09_29_35_DTL.log",
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/res002_seed2/logger/2026_07_17-12_05_25_DTL.log",
            
            # 7x28
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0_learned64_residual002/logger/2026_07_01-19_59_10_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed1_learned64_residual002/logger/2026_07_02-06_56_28_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed2_learned64_residual002/logger/2026_07_02-18_06_50_DTL.log",
        ],
    ),
    SeedGroup(
        key="residual002_head",
        label="outputLayer_residual",#"HyperLightMAPPO+head_residual",
        paths=[
            # grid4x4
            # 4x4
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/hyperlight_head_res002_seed0/logger/2026_07_21-16_23_24_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/hyperlight_head_res002_seed1/logger/2026_07_21-17_39_09_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow4x4/hyperlight_head_res002_seed2/logger/2026_07_21-18_56_18_DTL.log",
            
            #16x3
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperlight_head_res002_seed0/logger/2026_07_20-10_54_31_DTL.log",
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperlight_head_res002_seed1/logger/2026_07_20-13_41_16_DTL.log",
            "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperlight_head_res002_seed2/logger/2026_07_20-16_41_00_DTL.log",
            
            # 7x28
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed0_learned64_head_s002/logger/2026_07_06-18_33_38_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed1_learned64_head_s002/logger/2026_07_07-05_51_17_DTL.log",
            # "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow7x28/seed2_learned64_head_s002/logger/2026_07_07-17_17_54_DTL.log",
        ],
    ),
    
    # SeedGroup(
    #     key="MAPPO_IRU_n1_one_hot(id)",
    #     label="MAPPO+IRU_n1_one_hot(id)",
    #     paths=[
    #         # grid4x4
    #         # 4x4
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow4x4/mappo_iru_both_n1_seed0/logger/2026_07_10-17_38_04_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow4x4/mappo_iru_both_n1_seed1/logger/2026_07_10-22_52_38_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow4x4/mappo_iru_both_n1_seed2/logger/2026_07_10-23_59_06_DTL.log",
    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow7x28/seed0_iru1_one_hot(id)/logger/2026_07_13-01_23_07_DTL.log",
    #         # "",
    #         # "",
    #     ],
    # ),
    # SeedGroup(
    #     key="MAPPO_IRU_n2_one_hot(id)",
    #     label="MAPPO+IRU_n2_one_hot(id)",
    #     paths=[
    #         # grid4x4
    #         # 4x4
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow4x4/mappo_iru_both_n2_seed0/logger/2026_07_10-18_46_36_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow4x4/mappo_iru_both_n2_seed1/logger/2026_07_11-01_05_50_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow4x4/mappo_iru_both_n2_seed2/logger/2026_07_11-02_12_47_DTL.log",
    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow7x28/seed0_iru2_one_hot(id)/logger/2026_07_13-11_29_38_DTL.log",
    #         "",
    #         "",
    #     ],
    # ),
    # SeedGroup(
    #     key="MAPPO_IRU_n5_one_hot(id)",
    #     label="MAPPO+IRU_n5_one_hot(id)",
    #     paths=[
    #         # grid4x4
    #         # 4x4
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow4x4/mappo_iru_both_n5_seed0/logger/2026_07_10-19_55_27_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow4x4/mappo_iru_both_n5_seed1/logger/2026_07_11-03_19_43_DTL.log",
    #         # "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow4x4/mappo_iru_both_n5_seed2/logger/2026_07_11-04_28_41_DTL.log",
    #         # 7x28
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow7x28/seed0_iru5_one_hot(id)/logger/2026_07_13-22_05_49_DTL.log",
    #         # "",
    #         # "",
    #     ],
    # ),
    # SeedGroup(
    #     key="MAPPO+IRUequallearned(id)",          # Equal to native MAPPO with embedding learned 64
    #     label="MAPPO+IRUequallearned(id)",
    #     paths=[
    #         # grid4x4
    #         # 4x4
    #         # "",
    #         # "",
    #         # "",
    #         # 16x3
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow16x3/seed0_mlp_learned_Equal2NativeMAPPO_ep250/logger/2026_07_14-16_20_10_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow16x3/seed1_mlp_learned_Equal2NativeMAPPO_ep250/logger/2026_07_15-01_47_09_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow16x3/seed2_mlp_learned_Equal2NativeMAPPO_ep250/logger/2026_07_15-02_44_13_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="IRU_actor_n1_learned(id)",
    #     label="MAPPO+IRU_actor_n1_learned(id)",
    #     paths=[
    #         # grid4x4
    #         # 4x4
    #         # "",
    #         # "",
    #         # "",
    #         # 16x3
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow16x3/seed0_actor_iru1_learned_ep250/logger/2026_07_14-18_17_32_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow16x3/seed1_actor_iru1_learned_ep250/logger/2026_07_15-03_39_53_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow16x3/seed2_actor_iru1_learned_ep250/logger/2026_07_15-04_33_35_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="MAPPO_IRU_actor_n5_learned(id)",
    #     label="MAPPO+IRU_actor_n5_learned(id)",
    #     paths=[
    #         # grid4x4
    #         # 4x4
    #         # "",
    #         # "",
    #         # "",
    #         # 16x3
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow16x3/seed0_actor_iru5_learned_ep250/logger/2026_07_14-20_15_36_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow16x3/seed1_actor_iru5_learned_ep250/logger/2026_07_15-05_27_34_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow16x3/seed2_actor_iru5_learned_ep250/logger/2026_07_15-06_22_25_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="MAPPO_IRU_both_n5_learned(id)",
    #     label="MAPPO+IRU_both_n5_learned(id)",
    #     paths=[
    #         # grid4x4
    #         # 4x4
    #         # "",
    #         # "",
    #         # "",
    #         # 16x3
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_mappo_iru/cityflow16x3/seed0_both_iru5_learned_ep100/logger/2026_07_14-22_11_18_DTL.log",
    #         # "",
    #         # "",
    #     ],
    # ),
    
    # SeedGroup(
    #     key="hyperIRU_shared_mlp",
    #     label="shared_mlp",
    #     paths=[
    #         # grid4x4
    #         # 4x4
    #         # "",
    #         # "",
    #         # "",
    #         # 16x3
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_shared_mlp_seed0/logger/2026_07_17-17_29_15_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_shared_mlp_seed1/logger/2026_07_17-19_52_19_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_shared_mlp_seed2/logger/2026_07_17-22_13_52_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="hyperIRU_shared_mlp_pm19528_seed0",
    #     label="shared_mlp_pm19528",
    #     paths=[
    #         # grid4x4

    #         # 4x4

    #         # 16x3
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_shared_mlp_pm19528_seed0/logger/2026_07_20-03_43_48_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_shared_mlp_pm19528_seed1/logger/2026_07_20-06_06_28_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_shared_mlp_pm19528_seed2/logger/2026_07_20-08_31_24_DTL.log",

    #         # 7x28
           
    #     ],
    # ),
    # SeedGroup(
    #     key="hyperIRU_film_mlp",
    #     label="film_mlp",
    #     paths=[
    #         # grid4x4
    #         # 4x4
    #         # "",
    #         # "",
    #         # "",
    #         # 16x3
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_film_mlp_seed0/logger/2026_07_18-00_35_15_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_film_mlp_seed1/logger/2026_07_18-03_00_23_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_film_mlp_seed2/logger/2026_07_18-05_21_10_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="hyperIRU_shared_iru1",
    #     label="Shared_iru1",
    #     paths=[
    #         # grid4x4
    #         # 4x4
    #         # "",
    #         # "",
    #         # "",
    #         # 16x3
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_shared_iru1_seed0/logger/2026_07_18-07_40_48_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_shared_iru1_seed1/logger/2026_07_18-10_00_24_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_shared_iru1_seed2/logger/2026_07_18-13_22_04_DTL.log",
    #     ],
    # ),
    # SeedGroup(
    #     key="hyperIRU_film_iru1",
    #     label="film_iru1",
    #     paths=[
    #         # grid4x4
    #         # 4x4
    #         # "",
    #         # "",
    #         # "",
    #         # 16x3
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_film_iru1_seed0/logger/2026_07_18-15_42_25_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_film_iru1_seed1/logger/2026_07_18-18_04_36_DTL.log",
    #         "/DaRL/LibSignal/data/output_data/tsc/cityflow_hyperlight_mappo/cityflow16x3/hyperIRU_film_iru1_seed2/logger/2026_07_18-20_30_08_DTL.log",
    #     ],
    # ),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def parse_line(line: str) -> Optional[Record]:
    parts = line.strip().split("\t")
    if len(parts) != 9:
        return None
    try:
        return Record(
            algo=parts[0],
            mode=parts[1],
            episode=int(parts[2]),
            travel_time=float(parts[3]),
            loss=float(parts[4]),
            reward=float(parts[5]),
            queue=float(parts[6]),
            delay=float(parts[7]),
            throughput=float(parts[8]),
        )
    except ValueError:
        return None


def load_log(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = parse_line(line)
            if record is not None:
                records.append(record)
    for metric in METRICS:
        nonfinite_count = sum(
            1 for record in records if not math.isfinite(metric_value(record, metric))
        )
        if nonfinite_count:
            print(
                f"[WARN] Ignoring {nonfinite_count} non-finite {metric} value(s) in {path}"
            )
    return records


def parse_final_result_line(line: str) -> Optional[Record]:
    match = FINAL_RESULT_RE.search(line)
    if match is None:
        return None
    try:
        return Record(
            algo="final_checkpoint",
            mode="TEST",
            episode=-1,
            travel_time=float(match.group(1)),
            loss=math.nan,
            reward=float(match.group(2)),
            queue=float(match.group(3)),
            delay=float(match.group(4)),
            throughput=float(match.group(5)),
        )
    except ValueError:
        return None


def candidate_brf_logs(dtl_path: Path) -> List[Path]:
    if "_DTL" not in dtl_path.name:
        return []
    exact_match = dtl_path.with_name(dtl_path.name.replace("_DTL", "_BRF", 1))
    return [exact_match] if exact_match.is_file() else []


def load_final_checkpoint_record(dtl_path: Path) -> Optional[Record]:
    for brf_path in candidate_brf_logs(dtl_path):
        final_record = None
        with brf_path.open("r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_final_result_line(line)
                if parsed is not None:
                    final_record = parsed
        if final_record is not None:
            return final_record
    return None


def find_all_dtl_logs(folder: Path) -> List[Path]:
    """All *_DTL*.log files under folder, sorted ascending by mtime (oldest first)."""
    pattern = str(folder / "**" / "*_DTL*.log")
    candidates = [Path(p) for p in glob.glob(pattern, recursive=True)]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates


def find_latest_dtl_log(folder: Path) -> Optional[Path]:
    candidates = find_all_dtl_logs(folder)
    return candidates[-1] if candidates else None


def _resolve_candidates(raw_path: str) -> List[Path]:
    path = Path(raw_path).expanduser()
    root = repo_root()
    raw_posix = raw_path.replace("\\", "/")

    candidates: List[Path] = []
    for prefix in ("/DaRL/LibSignal/", "/home/meathoo/tsc_darl2/LibSignal/"):
        if raw_posix.startswith(prefix):
            candidates.append(root / raw_posix[len(prefix) :])

    if path.is_absolute():
        candidates.append(path)
        as_posix = path.as_posix()
        for prefix in ("/DaRL/LibSignal/", "/home/meathoo/tsc_darl2/LibSignal/"):
            if as_posix.startswith(prefix):
                candidates.append(root / as_posix[len(prefix) :])
    else:
        candidates.append(root / path)
    return candidates


def resolve_path(raw_path: str) -> Optional[Path]:
    """Resolve raw_path to a single DTL log file (latest by mtime if a directory)."""
    for candidate in _resolve_candidates(raw_path):
        if candidate.is_file():
            return candidate if candidate.suffix == ".log" else None
        if candidate.is_dir():
            latest = find_latest_dtl_log(candidate)
            if latest is not None:
                return latest
    return None


def resolve_log_dir(raw_path: str) -> Optional[Path]:
    """Resolve raw_path to a directory containing DTL logs, if it is one."""
    for candidate in _resolve_candidates(raw_path):
        if candidate.is_dir() and find_all_dtl_logs(candidate):
            return candidate
    return None


def load_log_dir_merged(folder: Path) -> List[Record]:
    """
    Merge every *_DTL*.log in a run folder into one continuous trajectory.

    A crashed-and-resumed run leaves multiple DTL logs behind (one per attempt),
    each covering a different episode range with possible overlap at the resume
    point. Taking only the newest file (by mtime) silently drops the episodes
    from before the last crash, which quietly corrupts whole-trajectory stats
    like "mean" even though tail-end stats like "last"/"best" happen to still
    look fine. Here we replay every file in mtime order and let a later write
    win for any (mode, episode) key that appears more than once, so the merged
    trajectory reflects the actual full run.
    """
    by_key: Dict[Tuple[str, int], Record] = {}
    for log_path in find_all_dtl_logs(folder):
        for record in load_log(log_path):
            by_key[(record.mode, record.episode)] = record
    mode_order = {"TRAIN": 0, "TEST": 1}
    return sorted(
        by_key.values(),
        key=lambda r: (r.episode, mode_order.get(r.mode, 2)),
    )


def metric_value(record: Record, metric: str) -> float:
    return float(getattr(record, metric))


def parse_seed_id(path: Path) -> Optional[str]:
    """Return a canonical numeric seed ID parsed from the nearest path component."""
    for part in reversed(path.parts):
        matches = list(SEED_ID_RE.finditer(part))
        if matches:
            return str(int(matches[-1].group(1)))
    return None


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < window:
        return values

    smoothed: List[float] = []
    left = (window - 1) // 2
    for idx in range(len(values)):
        start = min(max(0, idx - left), len(values) - window)
        end = start + window
        chunk = values[start:end]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def std_value(values: Sequence[float], ddof: int) -> float:
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    if len(values) <= ddof:
        return math.nan

    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - ddof)
    return math.sqrt(variance)


def format_float(value: float, decimals: int) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{decimals}f}"


def format_mean_std(mean_value: float, std: float, decimals: int) -> str:
    if not math.isfinite(mean_value) or not math.isfinite(std):
        return ""
    return f"{mean_value:.{decimals}f} +/- {std:.{decimals}f}"


def metric_is_smaller_better(metric: str) -> bool:
    return metric in {"travel_time", "loss", "queue", "delay"}


def finite_metric_records(
    records: Sequence[Record],
    mode: str,
    metric: str,
    episode_start: Optional[int],
    episode_end: Optional[int],
) -> List[Record]:
    filtered = [record for record in records if record.mode == mode]
    filtered.sort(key=lambda record: record.episode)
    if episode_start is not None:
        filtered = [record for record in filtered if record.episode >= episode_start]
    if episode_end is not None:
        filtered = [record for record in filtered if record.episode <= episode_end]
    return [record for record in filtered if math.isfinite(metric_value(record, metric))]


def seed_series(
    records: List[Record],
    mode: str,
    metric: str,
    episode_start: Optional[int],
    episode_end: Optional[int],
    ma_window: int,
) -> Dict[int, float]:
    filtered = finite_metric_records(
        records, mode, metric, episode_start, episode_end
    )

    episodes = [r.episode for r in filtered]
    values = [metric_value(r, metric) for r in filtered]
    values = moving_average(values, ma_window)
    return dict(zip(episodes, values))


def seed_metric_values(
    records: List[Record],
    mode: str,
    metric: str,
    episode_start: Optional[int],
    episode_end: Optional[int],
    ma_window: int,
) -> List[float]:
    filtered = finite_metric_records(
        records, mode, metric, episode_start, episode_end
    )

    values = [metric_value(r, metric) for r in filtered]
    return moving_average(values, ma_window)


def seed_checkpoint_record(
    records: List[Record],
    mode: str,
    episode_start: Optional[int],
    episode_end: Optional[int],
) -> Optional[Record]:
    filtered = finite_metric_records(
        records, mode, "travel_time", episode_start, episode_end
    )
    if not filtered:
        return None

    return min(filtered, key=lambda r: r.travel_time)


def seed_table_stat(values: Sequence[float], metric: str, statistic: str) -> Optional[float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return None
    if statistic == "last":
        return finite_values[-1]
    if statistic == "best":
        return min(finite_values) if metric_is_smaller_better(metric) else max(finite_values)
    if statistic == "mean":
        return sum(finite_values) / len(finite_values)
    if statistic == "tail5":
        tail_values = finite_values[-5:]
        return sum(tail_values) / len(tail_values)
    raise ValueError(f"Unknown table statistic: {statistic}")


def seed_table_value(
    seed: LoadedSeed,
    mode: str,
    metric: str,
    statistic: str,
    episode_start: Optional[int],
    episode_end: Optional[int],
    table_ma_window: int,
) -> Optional[float]:
    if statistic == "best_checkpoint":
        checkpoint = seed_checkpoint_record(seed.records, mode, episode_start, episode_end)
        if checkpoint is None:
            return None
        value = metric_value(checkpoint, metric)
        return value if math.isfinite(value) else None
    if statistic == "final_checkpoint":
        if mode != "TEST" or seed.final_record is None:
            return None
        value = metric_value(seed.final_record, metric)
        return value if math.isfinite(value) else None

    values = seed_metric_values(
        records=seed.records,
        mode=mode,
        metric=metric,
        episode_start=episode_start,
        episode_end=episode_end,
        ma_window=table_ma_window,
    )
    return seed_table_stat(values, metric, statistic)


def _finite_seed_table_values(
    seeds: Sequence[LoadedSeed],
    mode: str,
    metric: str,
    statistic: str,
    episode_start: Optional[int],
    episode_end: Optional[int],
    table_ma_window: int,
) -> List[float]:
    values: List[float] = []
    for seed in seeds:
        value = seed_table_value(
            seed=seed,
            mode=mode,
            metric=metric,
            statistic=statistic,
            episode_start=episode_start,
            episode_end=episode_end,
            table_ma_window=table_ma_window,
        )
        if value is not None and math.isfinite(value):
            values.append(value)
    return values


def _relative_improvement_percent(metric: str, baseline: float, candidate: float) -> float:
    if abs(baseline) < 1e-12:
        return math.nan
    if metric_is_smaller_better(metric):
        return (baseline - candidate) / abs(baseline) * 100.0
    return (candidate - baseline) / abs(baseline) * 100.0


def _candidate_wins(metric: str, baseline: float, candidate: float) -> bool:
    return candidate < baseline if metric_is_smaller_better(metric) else candidate > baseline


def _unique_seed_map(
    seeds: Sequence[LoadedSeed], side: str
) -> Dict[str, LoadedSeed]:
    by_id: Dict[str, LoadedSeed] = {}
    ambiguous_ids = set()
    for seed in seeds:
        seed_id = seed.seed_id or parse_seed_id(seed.log_path)
        if seed_id is None:
            print(
                f"[WARN] Excluding {side} log from pairwise comparison because "
                f"its seed ID cannot be parsed: {seed.log_path}"
            )
            continue
        if seed_id in ambiguous_ids:
            continue
        if seed_id in by_id:
            print(
                f"[WARN] Excluding duplicate {side} seed {seed_id} from pairwise "
                "comparison; the mapping is ambiguous."
            )
            by_id.pop(seed_id)
            ambiguous_ids.add(seed_id)
            continue
        by_id[seed_id] = seed
    return by_id


def _matched_seed_values(
    baseline_by_id: Dict[str, LoadedSeed],
    candidate_by_id: Dict[str, LoadedSeed],
    seed_ids: Sequence[str],
    mode: str,
    metric: str,
    statistic: str,
    episode_start: Optional[int],
    episode_end: Optional[int],
    table_ma_window: int,
) -> List[Tuple[str, float, float]]:
    pairs: List[Tuple[str, float, float]] = []
    for seed_id in seed_ids:
        baseline_value = seed_table_value(
            baseline_by_id[seed_id],
            mode,
            metric,
            statistic,
            episode_start,
            episode_end,
            table_ma_window,
        )
        candidate_value = seed_table_value(
            candidate_by_id[seed_id],
            mode,
            metric,
            statistic,
            episode_start,
            episode_end,
            table_ma_window,
        )
        if (
            baseline_value is None
            or candidate_value is None
            or not math.isfinite(baseline_value)
            or not math.isfinite(candidate_value)
        ):
            continue
        pairs.append((seed_id, baseline_value, candidate_value))
    return pairs


def _print_aligned_rows(rows: Sequence[Dict[str, str]], columns: Sequence[Tuple[str, str, str]]) -> None:
    widths = {
        key: max(len(label), *(len(row[key]) for row in rows))
        for key, label, _ in columns
    }

    def format_cell(value: str, key: str, align: str) -> str:
        if align == "right":
            return value.rjust(widths[key])
        return value.ljust(widths[key])

    header = "  ".join(
        format_cell(label, key, align)
        for key, label, align in columns
    )
    separator = "  ".join("-" * widths[key] for key, _, _ in columns)
    print(header)
    print(separator)
    for row in rows:
        print(
            "  ".join(
                format_cell(row[key], key, align)
                for key, _, align in columns
            )
        )


def _build_pairwise_comparison_rows(
    baseline_seeds: Sequence[LoadedSeed],
    candidate_seeds: Sequence[LoadedSeed],
    modes: Sequence[str],
    metrics: Sequence[str],
    comparison_statistics: Sequence[str],
    episode_start: Optional[int],
    episode_end: Optional[int],
    table_ma_window: int,
    decimals: int,
) -> List[Dict[str, str]]:
    comparison_rows: List[Dict[str, str]] = []
    baseline_by_id = _unique_seed_map(baseline_seeds, "baseline")
    candidate_by_id = _unique_seed_map(candidate_seeds, "candidate")
    shared_seed_ids = sorted(
        set(baseline_by_id).intersection(candidate_by_id), key=lambda value: int(value)
    )
    if not shared_seed_ids:
        print(
            "[WARN] Skipping pairwise comparison: baseline and candidate have "
            "no unambiguous matching seed IDs."
        )
        return comparison_rows

    for mode in modes:
        for metric in metrics:
            for statistic in comparison_statistics:
                matched_values = _matched_seed_values(
                    baseline_by_id,
                    candidate_by_id,
                    shared_seed_ids,
                    mode,
                    metric,
                    statistic,
                    episode_start,
                    episode_end,
                    table_ma_window,
                )
                if not matched_values:
                    print(
                        f"[WARN] Skipping {mode}/{metric}/{statistic}: no finite "
                        "values exist for the shared seed IDs."
                    )
                    continue

                pair_count = len(matched_values)
                wins = sum(
                    1
                    for _seed_id, baseline_value, candidate_value in matched_values
                    if _candidate_wins(metric, baseline_value, candidate_value)
                )
                baseline_values = [value[1] for value in matched_values]
                candidate_values = [value[2] for value in matched_values]
                baseline_mean = sum(baseline_values) / len(baseline_values)
                candidate_mean = sum(candidate_values) / len(candidate_values)
                improvement = _relative_improvement_percent(metric, baseline_mean, candidate_mean)
                if statistic == "best":
                    baseline_drift = 0.0
                    candidate_drift = 0.0
                else:
                    best_values = _matched_seed_values(
                        baseline_by_id,
                        candidate_by_id,
                        [value[0] for value in matched_values],
                        mode,
                        metric,
                        "best",
                        episode_start,
                        episode_end,
                        table_ma_window,
                    )
                    if not best_values:
                        baseline_drift = math.nan
                        candidate_drift = math.nan
                    else:
                        best_baseline_mean = sum(value[1] for value in best_values) / len(best_values)
                        best_candidate_mean = sum(value[2] for value in best_values) / len(best_values)
                        if metric_is_smaller_better(metric):
                            baseline_drift = baseline_mean - best_baseline_mean
                            candidate_drift = candidate_mean - best_candidate_mean
                        else:
                            baseline_drift = best_baseline_mean - baseline_mean
                            candidate_drift = best_candidate_mean - candidate_mean
                drift_delta = (
                    candidate_drift - baseline_drift
                    if math.isfinite(candidate_drift) and math.isfinite(baseline_drift)
                    else math.nan
                )

                comparison_rows.append(
                    {
                        "mode": mode,
                        "metric": metric,
                        "stat": statistic,
                        "baseline": format_float(baseline_mean, decimals),
                        "candidate": format_float(candidate_mean, decimals),
                        "improve_%": format_float(improvement, decimals),
                        "win_rate": f"{wins}/{pair_count}",
                        "seeds": ",".join(value[0] for value in matched_values),
                        "base_drift": format_float(baseline_drift, decimals),
                        "cand_drift": format_float(candidate_drift, decimals),
                        "drift_delta": format_float(drift_delta, decimals),
                    }
                )
    return comparison_rows


def print_baseline_candidate_comparisons(
    loaded: Dict[str, List[LoadedSeed]],
    modes: Sequence[str],
    metrics: Sequence[str],
    statistics: Sequence[str],
    episode_start: Optional[int],
    episode_end: Optional[int],
    table_ma_window: int,
    decimals: int,
) -> None:
    if len(loaded) < 2:
        return

    group_items = list(loaded.items())
    baseline_key, baseline_seeds = group_items[0]
    baseline_label = baseline_seeds[0].group_label if baseline_seeds else baseline_key
    comparison_statistics = [
        statistic
        for statistic in ("tail5", "last", "best")
        if statistic in statistics
    ]
    if not comparison_statistics:
        comparison_statistics = list(statistics[:1])

    print("")
    print(f"[COMPARE] Baseline: {baseline_label} ({baseline_key})")
    columns = [
        ("mode", "mode", "left"),
        ("metric", "metric", "left"),
        ("stat", "stat", "left"),
        ("baseline", "baseline", "right"),
        ("candidate", "candidate", "right"),
        ("improve_%", "improve_%", "right"),
        ("win_rate", "win_rate", "right"),
        ("seeds", "seeds", "right"),
        ("base_drift", "base_drift", "right"),
        ("cand_drift", "cand_drift", "right"),
        ("drift_delta", "drift_delta", "right"),
    ]
    for candidate_key, candidate_seeds in group_items[1:]:
        candidate_label = candidate_seeds[0].group_label if candidate_seeds else candidate_key
        print("")
        print(f"[COMPARE] Candidate: {candidate_label} ({candidate_key}) vs baseline")
        comparison_rows = _build_pairwise_comparison_rows(
            baseline_seeds=baseline_seeds,
            candidate_seeds=candidate_seeds,
            modes=modes,
            metrics=metrics,
            comparison_statistics=comparison_statistics,
            episode_start=episode_start,
            episode_end=episode_end,
            table_ma_window=table_ma_window,
            decimals=decimals,
        )
        if comparison_rows:
            _print_aligned_rows(comparison_rows, columns)
        else:
            print("No comparable rows.")


def aggregate_series(
    seeds: Sequence[LoadedSeed],
    mode: str,
    metric: str,
    episode_start: Optional[int],
    episode_end: Optional[int],
    ma_window: int,
    episode_policy: str,
) -> Optional[AvgSeries]:
    per_seed = [
        seed_series(seed.records, mode, metric, episode_start, episode_end, ma_window)
        for seed in seeds
    ]
    per_seed = [series for series in per_seed if series]
    if not per_seed:
        return None

    episode_sets = [set(series) for series in per_seed]
    if episode_policy == "union":
        episodes = sorted(set().union(*episode_sets))
    else:
        episodes = sorted(set.intersection(*episode_sets))
    if not episodes:
        return None

    means: List[float] = []
    stds: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []
    counts: List[int] = []

    for episode in episodes:
        values = [series[episode] for series in per_seed if episode in series]
        avg = sum(values) / len(values)
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        means.append(avg)
        stds.append(math.sqrt(variance))
        mins.append(min(values))
        maxs.append(max(values))
        counts.append(len(values))

    first_seed = seeds[0]
    return AvgSeries(
        group_key=first_seed.group_key,
        group_label=first_seed.group_label,
        mode=mode,
        metric=metric,
        episodes=episodes,
        mean=means,
        std=stds,
        min_value=mins,
        max_value=maxs,
        seed_count=counts,
    )


def group_by_key(groups: Iterable[SeedGroup]) -> Dict[str, SeedGroup]:
    return {group.key: group for group in groups}


def select_groups(group_args: Optional[List[str]]) -> List[SeedGroup]:
    available = group_by_key(SEED_GROUPS)
    if not group_args or "all" in group_args:
        return list(SEED_GROUPS)

    selected: List[SeedGroup] = []
    unknown: List[str] = []
    for key in group_args:
        group = available.get(key)
        if group is None:
            unknown.append(key)
        else:
            selected.append(group)

    if unknown:
        valid = ", ".join(["all"] + sorted(available))
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}. Valid choices: {valid}")
    return selected


def select_values(raw_values: Optional[List[str]], valid_values: Sequence[str], name: str) -> List[str]:
    if not raw_values or "all" in raw_values:
        return list(valid_values)
    invalid = [value for value in raw_values if value not in valid_values]
    if invalid:
        valid = ", ".join(["all"] + list(valid_values))
        raise ValueError(f"Unknown {name}: {', '.join(invalid)}. Valid choices: {valid}")
    return raw_values


def load_groups(groups: Sequence[SeedGroup], strict: bool) -> Dict[str, List[LoadedSeed]]:
    loaded: Dict[str, List[LoadedSeed]] = {}
    for group in groups:
        group_seeds: List[LoadedSeed] = []
        for idx, raw_path in enumerate(group.paths):
            log_dir = resolve_log_dir(raw_path)
            log_path = find_latest_dtl_log(log_dir) if log_dir is not None else resolve_path(raw_path)
            seed_id = parse_seed_id(log_path or Path(raw_path))
            seed_label = f"seed{seed_id}" if seed_id is not None else f"unidentified{idx}"
            if log_path is None:
                message = f"[WARN] Missing log for {group.key}/{seed_label}: {raw_path}"
                if strict:
                    raise FileNotFoundError(message)
                print(message)
                continue

            if seed_id is None:
                print(
                    f"[WARN] Seed ID cannot be parsed for {group.key}: {log_path}. "
                    "This log remains usable for aggregation but is excluded from pairwise comparisons."
                )

            all_logs = find_all_dtl_logs(log_dir) if log_dir is not None else [log_path]
            if len(all_logs) > 1:
                records = load_log_dir_merged(log_dir)
                merge_suffix = f", merged {len(all_logs)} logs"
            else:
                records = load_log(log_path)
                merge_suffix = ""
            if not records:
                message = f"[WARN] Empty or unparsable log for {group.key}/{seed_label}: {log_path}"
                if strict:
                    raise RuntimeError(message)
                print(message)
                continue

            final_record = load_final_checkpoint_record(log_path)
            final_suffix = ", final=loaded" if final_record is not None else ""
            print(
                f"[OK] {group.label}/{seed_label} <- {log_path} "
                f"(rows={len(records)}{merge_suffix}{final_suffix})"
            )
            group_seeds.append(
                LoadedSeed(
                    group_key=group.key,
                    group_label=group.label,
                    seed_label=seed_label,
                    log_path=log_path,
                    records=records,
                    final_record=final_record,
                    seed_id=seed_id,
                )
            )

        if group_seeds:
            loaded[group.key] = group_seeds
        elif strict:
            raise RuntimeError(f"No usable seeds loaded for group: {group.key}")
    return loaded


def build_all_series(
    loaded: Dict[str, List[LoadedSeed]],
    modes: Sequence[str],
    metrics: Sequence[str],
    episode_start: Optional[int],
    episode_end: Optional[int],
    ma_window: int,
    episode_policy: str,
) -> Dict[Tuple[str, str, str], AvgSeries]:
    result: Dict[Tuple[str, str, str], AvgSeries] = {}
    for group_key, seeds in loaded.items():
        for mode in modes:
            for metric in metrics:
                series = aggregate_series(
                    seeds=seeds,
                    mode=mode,
                    metric=metric,
                    episode_start=episode_start,
                    episode_end=episode_end,
                    ma_window=ma_window,
                    episode_policy=episode_policy,
                )
                if series is not None:
                    result[(group_key, mode, metric)] = series
    return result


def band_values(series: AvgSeries, band: str) -> Optional[Tuple[List[float], List[float]]]:
    if band == "none":
        return None
    if band == "minmax":
        return series.min_value, series.max_value
    lower = [avg - std for avg, std in zip(series.mean, series.std)]
    upper = [avg + std for avg, std in zip(series.mean, series.std)]
    return lower, upper


def apply_axis_style(ax, metric: str) -> None:
    ax.set_xlabel("Episode")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25, linestyle="--")
    if metric == "loss":
        ax.set_yscale("symlog", linthresh=1.0)


def plot_metric(
    series_by_key: Dict[Tuple[str, str, str], AvgSeries],
    loaded: Dict[str, List[LoadedSeed]],
    mode: str,
    metric: str,
    output_dir: Path,
    dpi: int,
    band: str,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    used = 0

    for group_key in loaded:
        series = series_by_key.get((group_key, mode, metric))
        if series is None:
            continue
        line = ax.plot(series.episodes, series.mean, linewidth=1.8, label=series.group_label)[0]
        values = band_values(series, band)
        if values is not None:
            lower, upper = values
            ax.fill_between(series.episodes, lower, upper, color=line.get_color(), alpha=0.14)
        used += 1

    if used == 0:
        plt.close(fig)
        return

    ax.set_title(f"{metric.upper()} ({mode}) seed average")
    apply_axis_style(ax, metric)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"avg_metric_{metric}_{mode}.png"
    fig.savefig(out_file, dpi=dpi)
    print(f"[DONE] Saved: {out_file}")
    if show:
        plt.show()
    plt.close(fig)


def plot_all_metrics(
    series_by_key: Dict[Tuple[str, str, str], AvgSeries],
    loaded: Dict[str, List[LoadedSeed]],
    mode: str,
    metrics: Sequence[str],
    output_dir: Path,
    dpi: int,
    band: str,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    any_data = False

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        if metric not in metrics:
            ax.set_title(f"{metric} (skipped)")
            ax.axis("off")
            continue

        used = 0
        for group_key in loaded:
            series = series_by_key.get((group_key, mode, metric))
            if series is None:
                continue
            line = ax.plot(series.episodes, series.mean, linewidth=1.3, label=series.group_label)[0]
            values = band_values(series, band)
            if values is not None:
                lower, upper = values
                ax.fill_between(series.episodes, lower, upper, color=line.get_color(), alpha=0.12)
            used += 1

        if used:
            any_data = True
            ax.set_title(metric)
            apply_axis_style(ax, metric)
        else:
            ax.set_title(f"{metric} (no data)")
            ax.axis("off")

    if not any_data:
        plt.close(fig)
        return

    handles, labels = [], []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))

    fig.suptitle(f"{mode} seed-average comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"avg_all_metrics_{mode}.png"
    fig.savefig(out_file, dpi=dpi)
    print(f"[DONE] Saved: {out_file}")
    if show:
        plt.show()
    plt.close(fig)


def write_summary_csv(
    series_by_key: Dict[Tuple[str, str, str], AvgSeries],
    loaded: Dict[str, List[LoadedSeed]],
    modes: Sequence[str],
    metrics: Sequence[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        summary_file = output_dir / f"avg_summary_{mode}.csv"
        series_file = output_dir / f"avg_series_{mode}.csv"

        with summary_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "group",
                    "metric",
                    "episodes",
                    "seed_count_min",
                    "seed_count_max",
                    "first_mean",
                    "last_mean",
                    "best_mean",
                    "best_episode",
                    "mean_of_mean",
                    "last_std",
                ]
            )

            for group_key in loaded:
                for metric in metrics:
                    series = series_by_key.get((group_key, mode, metric))
                    if series is None:
                        writer.writerow([group_key, metric, 0, "", "", "", "", "", "", "", ""])
                        continue

                    best_value = (
                        min(series.mean) if metric_is_smaller_better(metric) else max(series.mean)
                    )
                    best_index = series.mean.index(best_value)
                    writer.writerow(
                        [
                            series.group_label,
                            metric,
                            len(series.episodes),
                            min(series.seed_count),
                            max(series.seed_count),
                            series.mean[0],
                            series.mean[-1],
                            best_value,
                            series.episodes[best_index],
                            sum(series.mean) / len(series.mean),
                            series.std[-1],
                        ]
                    )
        print(f"[DONE] Saved: {summary_file}")

        with series_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["group", "metric", "episode", "mean", "std", "min", "max", "seed_count"])
            for group_key in loaded:
                for metric in metrics:
                    series = series_by_key.get((group_key, mode, metric))
                    if series is None:
                        continue
                    for idx, episode in enumerate(series.episodes):
                        writer.writerow(
                            [
                                series.group_label,
                                metric,
                                episode,
                                series.mean[idx],
                                series.std[idx],
                                series.min_value[idx],
                                series.max_value[idx],
                                series.seed_count[idx],
                            ]
                        )
        print(f"[DONE] Saved: {series_file}")


def write_paper_table_csv(
    loaded: Dict[str, List[LoadedSeed]],
    modes: Sequence[str],
    metrics: Sequence[str],
    output_dir: Path,
    episode_start: Optional[int],
    episode_end: Optional[int],
    statistics: Sequence[str],
    ddof: int,
    decimals: int,
    table_ma_window: int,
    min_seed_count: int = 2,
    strict_min_seed_count: bool = False,
) -> None:
    if min_seed_count < 1:
        raise ValueError("min_seed_count must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_ids = sorted(
        {
            seed.seed_id
            for seeds in loaded.values()
            for seed in seeds
            if seed.seed_id is not None
        },
        key=lambda value: int(value),
    )
    max_unidentified_count = max(
        (
            sum(seed.seed_id is None for seed in seeds)
            for seeds in loaded.values()
        ),
        default=0,
    )
    seed_columns = [f"seed_{seed_id}" for seed_id in seed_ids]
    seed_columns.extend(
        f"unidentified_seed_{idx}" for idx in range(max_unidentified_count)
    )

    for mode in modes:
        long_file = output_dir / f"paper_table_long_{mode}.csv"
        wide_file = output_dir / f"paper_table_wide_{mode}.csv"

        rows: List[Dict[str, object]] = []
        with long_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "group",
                    "mode",
                    "metric",
                    "statistic",
                    "seed_count",
                    "mean",
                    "std",
                    "mean_std",
                    "std_type",
                    "status",
                    "seed_ids_used",
                    *seed_columns,
                ]
            )

            for group_key, seeds in loaded.items():
                group_label = seeds[0].group_label if seeds else group_key
                for metric in metrics:
                    for statistic in statistics:
                        seed_values: List[float] = []
                        values_by_seed_id: Dict[str, float] = {}
                        unidentified_values: List[float] = []
                        for seed in seeds:
                            value = seed_table_value(
                                seed=seed,
                                mode=mode,
                                metric=metric,
                                statistic=statistic,
                                episode_start=episode_start,
                                episode_end=episode_end,
                                table_ma_window=table_ma_window,
                            )
                            if value is not None and math.isfinite(value):
                                seed_values.append(value)
                                if seed.seed_id is None:
                                    unidentified_values.append(value)
                                else:
                                    values_by_seed_id[seed.seed_id] = value

                        mean_value = sum(seed_values) / len(seed_values) if seed_values else math.nan
                        enough_seeds = len(seed_values) >= min_seed_count
                        computed_std = std_value(seed_values, ddof=ddof)
                        std = computed_std if enough_seeds else math.nan
                        mean_std = (
                            format_mean_std(mean_value, std, decimals)
                            if enough_seeds
                            else ""
                        )
                        std_type = "sample" if ddof == 1 else "population"
                        if not enough_seeds:
                            status = f"insufficient_seeds:{len(seed_values)}<{min_seed_count}"
                            message = (
                                f"[WARN] Paper-table row {group_label}/{mode}/{metric}/{statistic} "
                                f"has {len(seed_values)} finite seed(s); at least "
                                f"{min_seed_count} required. Mean/std cell left blank."
                            )
                            if strict_min_seed_count:
                                raise RuntimeError(message)
                            print(message)
                        elif not math.isfinite(computed_std):
                            status = f"std_undefined:n={len(seed_values)},ddof={ddof}"
                        else:
                            status = "ok"
                        formatted_seed_values = [
                            format_float(values_by_seed_id.get(seed_id, math.nan), decimals)
                            for seed_id in seed_ids
                        ]
                        formatted_seed_values.extend(
                            format_float(value, decimals) for value in unidentified_values
                        )
                        formatted_seed_values.extend(
                            [""] * (max_unidentified_count - len(unidentified_values))
                        )
                        used_seed_ids = ",".join(
                            sorted(values_by_seed_id, key=lambda value: int(value))
                        )

                        writer.writerow(
                            [
                                group_label,
                                mode,
                                metric,
                                statistic,
                                len(seed_values),
                                format_float(mean_value, decimals),
                                format_float(std, decimals),
                                mean_std,
                                std_type,
                                status,
                                used_seed_ids,
                                *formatted_seed_values,
                            ]
                        )
                        rows.append(
                            {
                                "group": group_label,
                                "mode": mode,
                                "metric": metric,
                                "statistic": statistic,
                                "mean_std": mean_std,
                            }
                        )
        print(f"[DONE] Saved: {long_file}")

        with wide_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["group", "mode", "statistic", *metrics])

            for group_key, seeds in loaded.items():
                group_label = seeds[0].group_label if seeds else group_key
                for statistic in statistics:
                    values_by_metric = {
                        row["metric"]: row["mean_std"]
                        for row in rows
                        if row["group"] == group_label and row["statistic"] == statistic
                    }
                    writer.writerow(
                        [
                            group_label,
                            mode,
                            statistic,
                            *[values_by_metric.get(metric, "") for metric in metrics],
                        ]
                    )
        print(f"[DONE] Saved: {wide_file}")


def print_groups() -> None:
    print("Available groups:")
    for group in SEED_GROUPS:
        print(f"  {group.key}: {group.label} ({len(group.paths)} seeds)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average multiple seed logs by selected groups and plot comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list-groups", action="store_true", help="Print available seed groups and exit.")
    parser.add_argument("--groups", nargs="+", default=["all"], help="Groups to plot. Use 'all' for every group.")
    parser.add_argument("--metrics", nargs="+", default=["all"], help="Metrics to plot. Use 'all' for every metric.")
    parser.add_argument("--modes", nargs="+", default=["TRAIN", "TEST"], help="Modes to plot.")
    parser.add_argument("--episode-start", type=int, default=0, help="First episode to include.")
    parser.add_argument("--episode-end", type=int, default=250, help="Last episode to include.")
    parser.add_argument("--ma-window", type=int, default=10, help="Moving-average window. Use 1 to disable.")
    parser.add_argument(
        "--episode-policy",
        choices=["intersection", "union"],
        default="intersection",
        help="How to align episodes across seeds.",
    )
    parser.add_argument(
        "--band",
        choices=["std", "minmax", "none"],
        default="std",
        help="Shaded band around each averaged line.",
    )
    parser.add_argument("--output-dir", default="avg_compare_outputs", help="Directory for plots and CSV files.")
    parser.add_argument("--dpi", type=int, default=170, help="Figure DPI.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively after saving.")
    parser.add_argument("--no-all-plot", action="store_true", help="Skip 2x3 all-metrics figures.")
    parser.add_argument("--no-individual-plots", action="store_true", help="Skip one-figure-per-metric plots.")
    parser.add_argument(
        "--no-paper-table",
        action="store_true",
        help="Skip CSV files for paper-table mean/std across seeds.",
    )
    parser.add_argument(
        "--table-statistics",
        nargs="+",
        choices=TABLE_STATISTICS,
        default=["last", "best", "mean", "tail5"],
        help=(
            "Per-seed statistics to aggregate across seeds for paper tables. "
            "tail5 averages the last 5 evaluation points after episode filtering; "
            "best_checkpoint uses the TEST row with the best travel_time; "
            "final_checkpoint uses the BRF final loaded-checkpoint result."
        ),
    )
    parser.add_argument(
        "--table-std",
        choices=["sample", "population"],
        default="sample",
        help="Standard deviation type used in paper-table CSV files.",
    )
    parser.add_argument("--table-decimals", type=int, default=4, help="Decimals in paper-table CSV values.")
    parser.add_argument(
        "--table-use-smoothed",
        action="store_true",
        help="Use the moving-average series for paper-table values. Default uses raw log values.",
    )
    parser.add_argument(
        "--table-min-seeds",
        type=int,
        default=2,
        help="Minimum finite seed count required to emit a paper-table mean/std cell.",
    )
    parser.add_argument(
        "--table-strict-min-seeds",
        action="store_true",
        help="Fail instead of warning when a paper-table row has too few finite seeds.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on missing or empty seed logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_groups:
        print_groups()
        return

    groups = select_groups(args.groups)
    metrics = select_values(args.metrics, METRICS, "metric")
    modes = select_values(args.modes, MODES, "mode")
    output_dir = Path(args.output_dir)

    loaded = load_groups(groups, strict=args.strict)
    if not loaded:
        raise RuntimeError("No usable seed logs were loaded.")

    series_by_key = build_all_series(
        loaded=loaded,
        modes=modes,
        metrics=metrics,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        ma_window=args.ma_window,
        episode_policy=args.episode_policy,
    )
    if not series_by_key:
        raise RuntimeError("No averaged series could be built for the selected inputs.")

    for mode in modes:
        if not args.no_all_plot:
            plot_all_metrics(
                series_by_key=series_by_key,
                loaded=loaded,
                mode=mode,
                metrics=metrics,
                output_dir=output_dir,
                dpi=args.dpi,
                band=args.band,
                show=args.show,
            )
        if not args.no_individual_plots:
            for metric in metrics:
                plot_metric(
                    series_by_key=series_by_key,
                    loaded=loaded,
                    mode=mode,
                    metric=metric,
                    output_dir=output_dir,
                    dpi=args.dpi,
                    band=args.band,
                    show=args.show,
                )

    write_summary_csv(
        series_by_key=series_by_key,
        loaded=loaded,
        modes=modes,
        metrics=metrics,
        output_dir=output_dir,
    )
    if not args.no_paper_table:
        write_paper_table_csv(
            loaded=loaded,
            modes=modes,
            metrics=metrics,
            output_dir=output_dir,
            episode_start=args.episode_start,
            episode_end=args.episode_end,
            statistics=args.table_statistics,
            ddof=1 if args.table_std == "sample" else 0,
            decimals=args.table_decimals,
            table_ma_window=args.ma_window if args.table_use_smoothed else 1,
            min_seed_count=args.table_min_seeds,
            strict_min_seed_count=args.table_strict_min_seeds,
        )
        print_baseline_candidate_comparisons(
            loaded=loaded,
            modes=modes,
            metrics=metrics,
            statistics=args.table_statistics,
            episode_start=args.episode_start,
            episode_end=args.episode_end,
            table_ma_window=args.ma_window if args.table_use_smoothed else 1,
            decimals=args.table_decimals,
        )


if __name__ == "__main__":
    main()
