from collections import deque
import hashlib
import os

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from .actor import BaseActor
from .hyperlight_architecture import DirectedGraphCritic, MovementTokenEncoder
from .iru import IRUNetwork
from .hypernetwork import (
    build_generated_param_init_config,
    build_hypernetwork,
)
from .rl_agent import RLAgent
from . import utils
from common.registry import Registry
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator


@Registry.register_model('hyperlight_spo')
@Registry.register_model('hyperlight_ppo')
class HyperLightPPOAgent(RLAgent):
    """
    HyperMARL-style PPO/IPPO controller for TSC.

    The actor and value networks are generated from agent embeddings, following
    the public HyperMARL implementation style. Set centralized_critic=True for
    a MAPPO-style value input that includes global traffic context.
    """

    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])

        cfg = Registry.mapping['model_mapping']['setting'].param
        trainer_cfg = Registry.mapping['trainer_mapping']['setting'].param

        self.world = world
        self.rank = rank
        self.sub_agents = len(self.world.intersections)
        self.phase_lengths = np.asarray(
            [len(inter.phases) for inter in self.world.intersections],
            dtype=np.int64,
        )
        self.action_space = gym.spaces.Discrete(int(self.phase_lengths.max()))

        use_cuda = bool(cfg.get('use_cuda', True))
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

        self.phase = bool(cfg.get('phase', True))
        self.one_hot = bool(cfg.get('one_hot', True))
        self.vehicle_max = float(cfg.get('vehicle_max', 1.0))
        if self.vehicle_max <= 0:
            self.vehicle_max = 1.0
        state_features = cfg.get('state_features', ['lane_count', 'lane_waiting_count'])
        self.state_features = state_features if isinstance(state_features, list) else [state_features]

        self.gamma = float(cfg.get('gamma', 0.99))
        self.gae_lambda = float(cfg.get('gae_lambda', 0.95))
        self.clip_eps = float(cfg.get('clip_eps', 0.2))
        self.policy_objective = str(cfg.get('policy_objective', cfg.get('ppo_objective', 'ppo'))).lower()
        if self.policy_objective not in ('ppo', 'spo'):
            raise ValueError(f"Unknown HyperLight PPO policy_objective: {self.policy_objective}")
        self.spo_eps = float(cfg.get('spo_eps', self.clip_eps))
        if self.spo_eps <= 0.0:
            raise ValueError('spo_eps must be positive')
        self.clip_vf = cfg.get('clip_vf', 0.2)
        self.clip_vf = None if self.clip_vf is None else float(self.clip_vf)
        self.entropy_coef = float(cfg.get('entropy_coef', cfg.get('ent_coef', 0.01)))
        self.value_coef = float(cfg.get('value_coef', cfg.get('vf_coef', 0.5)))
        self.reward_scale = float(cfg.get('reward_scale', 1.0))
        self.reward_mode = str(cfg.get('reward_mode', 'queue')).lower()
        reward_mode_aliases = {
            'mean_waiting': 'queue',
            'waiting': 'queue',
            'mplight': 'queue',
            'pressure': 'pressure_abs',
        }
        self.reward_mode = reward_mode_aliases.get(self.reward_mode, self.reward_mode)
        if self.reward_mode not in ('queue', 'pressure_abs', 'queue_pressure'):
            raise ValueError(f"Unknown HyperLight PPO reward_mode: {self.reward_mode}")
        self.pressure_balance_coef = float(cfg.get('pressure_balance_coef', 0.2))
        self.pressure_release_coef = float(cfg.get('pressure_release_coef', 0.0))
        self.grad_clip = float(cfg.get('grad_clip', 0.5))
        self.ppo_epochs = max(1, int(cfg.get('ppo_epochs', 4)))
        self.ppo_rollout_steps = max(1, int(cfg.get('ppo_rollout_steps', 360)))
        self.ppo_minibatch_size = max(1, int(cfg.get('ppo_minibatch_size', 2048)))
        self.value_chunk_size = int(cfg.get('value_chunk_size', 0))
        self.test_action_mode = str(cfg.get('test_action_mode', 'argmax')).lower()
        if self.test_action_mode == 'stochastic':
            self.test_action_mode = 'sample'
        if self.test_action_mode not in ('argmax', 'sample'):
            raise ValueError(f"Unknown test_action_mode: {self.test_action_mode}")
        self.test_temperature = max(1e-6, float(cfg.get('test_temperature', 1.0)))
        self.normalize_advantage = bool(cfg.get('normalize_advantage', True))
        self.centralized_critic = bool(cfg.get('centralized_critic', False))
        self.centralized_critic_mode = str(cfg.get('centralized_critic_mode', 'pooled')).lower()
        if self.centralized_critic_mode in ('directed_graph', 'directed-graph', 'gat'):
            self.centralized_critic_mode = 'graph'
        if self.centralized_critic_mode == 'graph' and not self.centralized_critic:
            raise ValueError('centralized_critic_mode=graph requires centralized_critic=True')
        self.graph_critic_enabled = bool(
            self.centralized_critic and self.centralized_critic_mode == 'graph'
        )
        self.activation = str(cfg.get('activation', 'relu')).lower()
        if self.activation not in ('relu', 'tanh'):
            raise ValueError(f"Unknown HyperLight PPO activation: {self.activation}")

        self.movement_encoder_enabled = bool(cfg.get('movement_encoder_enabled', False))
        self.movement_token_dim = int(cfg.get('movement_token_dim', 64))
        self.movement_encoder_dim = int(cfg.get('movement_encoder_dim', self.movement_token_dim))
        self.movement_encoder_heads = int(cfg.get('movement_encoder_heads', 4))
        self.movement_encoder_layers = int(cfg.get('movement_encoder_layers', 1))
        self.movement_encoder_ff_dim = int(
            cfg.get('movement_encoder_ff_dim', 2 * self.movement_token_dim)
        )
        self.movement_encoder_dropout = float(cfg.get('movement_encoder_dropout', 0.0))

        self.graph_critic_hidden_dim = int(cfg.get('graph_critic_hidden_dim', 128))
        self.graph_critic_layers = int(cfg.get('graph_critic_layers', 2))
        self.graph_critic_heads = int(cfg.get('graph_critic_heads', 4))
        self.graph_critic_dropout = float(cfg.get('graph_critic_dropout', 0.0))
        self.graph_critic_use_edge_weight = bool(cfg.get('graph_critic_use_edge_weight', False))
        self.graph_critic_edge_weight_scale = float(cfg.get('graph_critic_edge_weight_scale', 1.0))
        self.graph_critic_global_pool = bool(cfg.get('graph_critic_global_pool', True))
        self.graph_critic_film = bool(cfg.get('graph_critic_film', True))
        graph_film_hidden = cfg.get('graph_critic_film_hidden', [64])
        self.graph_critic_film_hidden = self._as_int_list(graph_film_hidden)
        self.graph_critic_film_scale = float(cfg.get('graph_critic_film_scale', 0.1))
        self.graph_message_direction = str(
            cfg.get('graph_message_direction', 'downstream_to_upstream')
        ).lower()
        if self.movement_encoder_enabled and self.movement_encoder_dropout != 0.0:
            raise ValueError(
                'movement_encoder_dropout must be 0 for PPO so rollout and update '
                'log-probabilities use the same deterministic encoder'
            )
        if self.graph_critic_enabled and self.graph_critic_dropout != 0.0:
            raise ValueError(
                'graph_critic_dropout must be 0 for PPO value targets and updates to match'
            )

        self.actor_hidden1 = int(cfg.get('actor_hidden1', 64))
        self.actor_hidden2 = int(cfg.get('actor_hidden2', 64))
        self.actor_arch = str(cfg.get('hyper_actor_arch', 'mlp')).lower()
        actor_arch_aliases = {
            'shared_mlp': 'mlp',
            'shared_iru': 'iru',
            'interpolation_recurrent_unit': 'iru',
        }
        self.actor_arch = actor_arch_aliases.get(self.actor_arch, self.actor_arch)
        if self.actor_arch not in ('mlp', 'iru'):
            raise ValueError(f'Unknown HyperLight actor architecture: {self.actor_arch}')
        self.iru_hidden_dim = int(cfg.get('iru_hidden_dim', 64))
        self.iru_actor_hidden_dim = int(
            cfg.get('iru_actor_hidden_dim', self.iru_hidden_dim)
        )
        self.iru_num_blocks = int(cfg.get('iru_num_blocks', 1))
        self.iru_layer_norm = bool(cfg.get('iru_layer_norm', True))
        default_iru_steps = int(cfg.get('iru_steps', 1))
        self.iru_actor_steps = int(cfg.get('iru_actor_steps', default_iru_steps))
        value_hidden = cfg.get('value_hidden', cfg.get('critic_hidden', [64, 64]))
        if not isinstance(value_hidden, list):
            value_hidden = [int(value_hidden)]
        self.value_hidden = [int(item) for item in value_hidden]

        self.hypernet_type = cfg.get('actor_hypernet_type', cfg.get('hypernet_type', 'mlp'))
        self.value_hypernet_type = cfg.get(
            'value_hypernet_type',
            cfg.get('critic_hypernet_type', self.hypernet_type),
        )
        self.hyper_head_mode = str(cfg.get('hyper_head_mode', 'layerwise')).lower()
        self.hyper_use_bias = bool(cfg.get('hyper_use_bias', True))
        self.hyper_head_init_gain = float(cfg.get('hyper_head_init_gain', 1.0))
        self.hyper_chunk_size = int(cfg.get('hyper_chunk_size', 8))
        self.hyper_chunk_embed_dim = int(cfg.get('hyper_chunk_embed_dim', 16))
        if self.hyper_chunk_size <= 0:
            raise ValueError('hyper_chunk_size must be positive')
        if self.hyper_chunk_embed_dim <= 0:
            raise ValueError('hyper_chunk_embed_dim must be positive')
        self.hyper_adapter_mode = str(cfg.get('hyper_adapter_mode', 'generated')).lower()
        adapter_aliases = {
            'full': 'generated',
            'dense': 'generated',
            'weight': 'generated',
            'weight_generation': 'generated',
        }
        self.hyper_adapter_mode = adapter_aliases.get(
            self.hyper_adapter_mode,
            self.hyper_adapter_mode,
        )
        if self.hyper_adapter_mode not in ('generated', 'film', 'none'):
            raise ValueError(f"Unknown hyper_adapter_mode: {self.hyper_adapter_mode}")
        self.hyper_critic_adapter_mode = str(
            cfg.get('hyper_critic_adapter_mode', 'generated')
        ).lower()
        critic_adapter_aliases = {
            'full': 'generated',
            'dense': 'generated',
            'weight': 'generated',
            'weight_generation': 'generated',
        }
        self.hyper_critic_adapter_mode = critic_adapter_aliases.get(
            self.hyper_critic_adapter_mode,
            self.hyper_critic_adapter_mode,
        )
        if self.hyper_critic_adapter_mode not in ('generated', 'film'):
            raise ValueError(
                f"Unknown hyper_critic_adapter_mode: {self.hyper_critic_adapter_mode}"
            )
        self.hyper_film_scale = float(cfg.get('hyper_film_scale', 0.1))
        self.hyper_film_init_zero = bool(cfg.get('hyper_film_init_zero', True))
        if self.hyper_film_scale < 0.0:
            raise ValueError('hyper_film_scale must be non-negative')
        self.hyper_residual = bool(cfg.get('hyper_residual', cfg.get('hyper_residual_enabled', False)))
        self.hyper_residual_mode = str(cfg.get('hyper_residual_mode', 'full')).lower()
        if self.hyper_residual_mode in ('low_rank', 'low-rank'):
            self.hyper_residual_mode = 'lora'
        if self.hyper_residual_mode in ('head_only', 'head-only', 'last_layer', 'last-layer'):
            self.hyper_residual_mode = 'head'
        if self.hyper_residual_mode not in ('full', 'lora', 'head'):
            raise ValueError(f"Unknown hyper_residual_mode: {self.hyper_residual_mode}")
        residual_scale = float(cfg.get('hyper_residual_scale', 0.02))
        actor_residual_scale = cfg.get('hyper_residual_actor_scale', None)
        value_residual_scale = cfg.get('hyper_residual_value_scale', None)
        self.hyper_residual_actor_scale = float(
            residual_scale if actor_residual_scale is None else actor_residual_scale
        )
        self.hyper_residual_value_scale = float(
            residual_scale if value_residual_scale is None else value_residual_scale
        )
        if self.hyper_residual_actor_scale < 0.0 or self.hyper_residual_value_scale < 0.0:
            raise ValueError('hyper residual scales must be non-negative')
        self.hyper_residual_log_diagnostics = bool(cfg.get('hyper_residual_log_diagnostics', True))
        lora_rank = int(cfg.get('hyper_lora_rank', 4))
        actor_lora_rank = cfg.get('hyper_lora_actor_rank', None)
        value_lora_rank = cfg.get('hyper_lora_value_rank', None)
        self.hyper_lora_actor_rank = int(lora_rank if actor_lora_rank is None else actor_lora_rank)
        self.hyper_lora_value_rank = int(lora_rank if value_lora_rank is None else value_lora_rank)
        if self.hyper_lora_actor_rank <= 0 or self.hyper_lora_value_rank <= 0:
            raise ValueError('hyper LoRA ranks must be positive')
        self.hyper_lora_bias = bool(cfg.get('hyper_lora_bias', False))
        if self.hyper_adapter_mode == 'film' and self.hyper_residual:
            raise ValueError(
                'hyper_adapter_mode=film and hyper_residual are alternative adapter paths; '
                'disable hyper_residual (LoRA/full/head) when using FiLM'
            )
        if self.hyper_adapter_mode == 'none' and self.hyper_residual:
            raise ValueError(
                'hyper_adapter_mode=none disables actor adaptation and cannot be '
                'combined with hyper_residual'
            )
        if self.hyper_critic_adapter_mode == 'film' and self.hyper_residual:
            raise ValueError(
                'hyper_critic_adapter_mode=film and hyper_residual are alternative '
                'critic adapter paths; disable hyper_residual when using critic FiLM'
            )
        if self.graph_critic_enabled and self.hyper_critic_adapter_mode == 'film':
            raise ValueError(
                'hyper_critic_adapter_mode=film applies to the pooled/flat critic; '
                'use graph_critic_film for a graph critic'
            )
        if (
            self.actor_arch == 'iru'
            and self.hyper_adapter_mode == 'generated'
            and bool(cfg.get('hyper_residual', False))
        ):
            raise ValueError(
                'Generated IRU actor weights do not yet support hyper_residual=True: '
                'the generated layout excludes IRU LayerNorm affine params (they stay '
                'shared/trainable), so it does not align with the residual base-module '
                'flattening used by hyper_residual. Use hyper_residual=False, or the '
                'film/none adapters, with an IRU actor.'
            )
        self.actor_rf_init_config = build_generated_param_init_config(
            cfg,
            output_gain_key='hyper_rf_actor_output_gain',
            default_output_gain=0.01,
        )
        self.value_rf_init_config = build_generated_param_init_config(
            cfg,
            output_gain_key='hyper_rf_value_output_gain',
            default_output_gain=1.0,
        )
        hyper_hidden = cfg.get('hyper_hidden', [64])
        if not isinstance(hyper_hidden, list):
            hyper_hidden = [int(hyper_hidden)]
        value_hyper_hidden = cfg.get('value_hyper_hidden', cfg.get('critic_hyper_hidden', hyper_hidden))
        if not isinstance(value_hyper_hidden, list):
            value_hyper_hidden = [int(value_hyper_hidden)]
        hyper_dropout = float(cfg.get('hyper_dropout', 0.0))
        if (
            self.hyper_adapter_mode == 'film'
            or self.hyper_critic_adapter_mode == 'film'
        ) and hyper_dropout != 0.0:
            raise ValueError(
                'hyper_dropout must be 0 for FiLM PPO so rollout and update outputs match'
            )

        self._build_generators()
        self.state_dim = self.ob_length
        if self.phase:
            self.state_dim += self.action_space.n if self.one_hot else 1
        self.raw_state_dim = self.state_dim
        self.movement_encoder = None
        self.movement_feature_indices = None
        self.movement_token_mask = None
        self.movement_phase_availability = None
        self.movement_source_position = None
        self.movement_position = None
        self.movement_turn_features = None
        if self.movement_encoder_enabled:
            self._build_movement_token_spec()
            self.movement_encoder = MovementTokenEncoder(
                len(self.state_features),
                self.action_space.n,
                token_dim=self.movement_token_dim,
                output_dim=self.movement_encoder_dim,
                num_heads=self.movement_encoder_heads,
                num_layers=self.movement_encoder_layers,
                feedforward_dim=self.movement_encoder_ff_dim,
                dropout=self.movement_encoder_dropout,
                static_feature_dim=4,
            ).to(self.device)
            self.policy_input_dim = self.movement_encoder_dim
        else:
            self.policy_input_dim = self.state_dim
        if not self.centralized_critic:
            self.value_input_dim = self.policy_input_dim
        elif self.centralized_critic_mode == 'concat':
            self.value_input_dim = self.policy_input_dim * self.sub_agents
        elif self.centralized_critic_mode == 'pooled':
            self.value_input_dim = self.policy_input_dim * 5
        elif self.centralized_critic_mode == 'graph':
            self.value_input_dim = self.policy_input_dim
        else:
            raise ValueError(f"Unknown centralized_critic_mode: {self.centralized_critic_mode}")
        self.action_mask = self._build_action_mask().to(self.device)

        raw_embedding_mode = str(cfg.get('agent_embedding_mode', 'one_hot')).lower()
        self.topology_aware_embedding = bool(cfg.get('topology_aware_embedding', False))
        if raw_embedding_mode in ('topology', 'learned_topology', 'topology_aware'):
            self.embedding_mode = 'learned'
            self.topology_aware_embedding = True
        elif raw_embedding_mode == 'one_hot_topology':
            self.embedding_mode = 'one_hot'
            self.topology_aware_embedding = True
        else:
            self.embedding_mode = raw_embedding_mode

        if self.embedding_mode == 'learned':
            embedding_dim = int(cfg.get('agent_embedding_dim', min(64, self.sub_agents)))
            self.agent_embeddings = nn.Parameter(torch.empty(self.sub_agents, embedding_dim, device=self.device))
            nn.init.orthogonal_(self.agent_embeddings)
            self.meta_dim = embedding_dim
        elif self.embedding_mode == 'one_hot':
            self.agent_embeddings = torch.eye(self.sub_agents, dtype=torch.float32, device=self.device)
            self.meta_dim = self.sub_agents
        else:
            raise ValueError(f"Unknown agent_embedding_mode: {self.embedding_mode}")

        self.topology_feature_names = []
        self.topology_encoder = None
        if self.topology_aware_embedding:
            topology_features, self.topology_feature_names = self._build_topology_features()
            self.registered_topology_features = torch.tensor(
                topology_features,
                dtype=torch.float32,
                device=self.device,
            )
            topology_hidden = int(cfg.get('topology_hidden_dim', 0))
            if topology_hidden > 0:
                self.topology_encoder = nn.Sequential(
                    nn.Linear(self.registered_topology_features.shape[-1], topology_hidden),
                    nn.ReLU(),
                    nn.Linear(topology_hidden, self.meta_dim),
                ).to(self.device)
            else:
                self.topology_encoder = nn.Linear(
                    self.registered_topology_features.shape[-1],
                    self.meta_dim,
                ).to(self.device)

        self.cos_enabled = bool(cfg.get('cos_enabled', False))
        self.cos_top_k = min(
            self.sub_agents,
            max(1, int(cfg.get('cos_top_k', min(5, self.sub_agents)))),
        )
        self.cos_feature_source = str(cfg.get('cos_feature_source', 'state_meta')).lower()
        if self.cos_feature_source not in ('state', 'meta', 'state_meta'):
            raise ValueError(f"Unknown cos_feature_source: {self.cos_feature_source}")
        self.cos_context_dim = int(cfg.get('cos_context_dim', self.meta_dim))
        self.cos_fusion_mode = str(cfg.get('cos_fusion_mode', 'concat')).lower()
        if self.cos_fusion_mode not in ('add', 'concat'):
            raise ValueError(f"Unknown cos_fusion_mode: {self.cos_fusion_mode}")
        self.cos_policy_coef = float(cfg.get('cos_policy_coef', 1.0))
        self.cos_entropy_coef = float(cfg.get('cos_entropy_coef', 0.0))
        self.cos_diag_coef = float(cfg.get('cos_diag_coef', 0.01))
        self.cos_symmetry_coef = float(cfg.get('cos_symmetry_coef', 0.01))
        self.cos_self_bias = float(cfg.get('cos_self_bias', 0.0))
        self.cos_logit_clip = float(cfg.get('cos_logit_clip', 0.0))
        self.cos_deterministic_eval = bool(cfg.get('cos_deterministic_eval', True))
        self.cos_log_diagnostics = bool(cfg.get('cos_log_diagnostics', True))
        self.cos_state_encoder = None
        self.cos_meta_encoder = None
        self.cos_selector = None
        self.cos_team_projector = None
        if self.cos_enabled:
            if self.cos_feature_source in ('state', 'state_meta'):
                self.cos_state_encoder = self._build_mlp(
                    self.state_dim,
                    self._as_int_list(cfg.get('cos_state_hidden', [128])),
                    self.cos_context_dim,
                    final_activation=True,
                ).to(self.device)
            if self.cos_feature_source in ('meta', 'state_meta'):
                self.cos_meta_encoder = nn.Linear(self.meta_dim, self.cos_context_dim).to(self.device)
            self.cos_selector = self._build_mlp(
                self.cos_context_dim,
                self._as_int_list(cfg.get('cos_selector_hidden', [64])),
                self.sub_agents,
                final_activation=False,
            ).to(self.device)
            if self.cos_fusion_mode == 'add':
                self.cos_team_projector = nn.Linear(self.cos_context_dim, self.meta_dim).to(self.device)
            else:
                self.cos_team_projector = self._build_mlp(
                    self.meta_dim + self.cos_context_dim,
                    self._as_int_list(cfg.get('cos_fusion_hidden', [])),
                    self.meta_dim,
                    final_activation=False,
                ).to(self.device)

        self.cos_pairwise_hops = None
        self.cos_pairwise_distances = None
        self._last_cos_diagnostics = {}
        self._cos_episode_diagnostics = []
        if self.cos_enabled:
            self.cos_pairwise_hops, self.cos_pairwise_distances = self._build_cos_pairwise_metrics()

        if self.actor_arch == 'iru':
            self.base_actor = IRUNetwork(
                self.policy_input_dim,
                self.iru_actor_hidden_dim,
                self.action_space.n,
                thinking_steps=self.iru_actor_steps,
                num_blocks=self.iru_num_blocks,
                layer_norm=self.iru_layer_norm,
            ).to(self.device)
        else:
            self.base_actor = BaseActor(
                self.policy_input_dim,
                self.actor_hidden1,
                self.actor_hidden2,
                self.action_space.n,
            ).to(self.device)
        self._iru_generated_actor = bool(
            self.actor_arch == 'iru' and self.hyper_adapter_mode == 'generated'
        )
        self.base_actor_trainable = bool(
            self.hyper_residual
            or self.hyper_adapter_mode in ('film', 'none')
            or self._iru_generated_actor
        )
        if self._iru_generated_actor:
            # Linear submodules (input_embedding, forget/input gates, output_head) are
            # hypernetwork-generated per agent every forward pass, so their base_actor
            # copy is just an unused shape template and must not receive gradients.
            # LayerNorm affine params have no (out_dim, in_dim) matrix shape the
            # chunked/layerwise heads can slice into, so they stay real, shared,
            # trainable parameters instead of being generated.
            for submodule in self.base_actor.modules():
                trainable = not isinstance(submodule, nn.Linear)
                for param in submodule.parameters(recurse=False):
                    param.requires_grad = trainable
        else:
            for param in self.base_actor.parameters():
                param.requires_grad = self.base_actor_trainable

        if self._iru_generated_actor:
            self.actor_layout = self._build_linear_layout_from_module(self.base_actor)
        else:
            self.actor_layout = self._build_layout_from_module(self.base_actor)
        self.actor_param_dim = self.actor_layout[-1][-1]
        self.actor_head_layout = None
        self.actor_head_param_dim = 0
        if self._use_head_residual():
            self.actor_head_layout, self.actor_head_param_dim = self._build_head_layout(
                self.actor_layout
            )
        self.actor_lora_layout = []
        self.actor_lora_param_dim = 0
        if self._use_lora_residual():
            self.actor_lora_layout, self.actor_lora_param_dim = self._build_lora_layout(
                self.actor_layout,
                self.hyper_lora_actor_rank,
                use_bias=self.hyper_lora_bias,
            )
        if self.actor_arch == 'iru':
            self.actor_film_param_dim = 4 * self.iru_actor_hidden_dim
        else:
            self.actor_film_param_dim = 2 * (self.actor_hidden1 + self.actor_hidden2)
        if self.hyper_adapter_mode == 'none':
            actor_hyper_output_dim = None
        elif self.hyper_adapter_mode == 'film':
            actor_hyper_output_dim = self.actor_film_param_dim
        elif self._use_lora_residual():
            actor_hyper_output_dim = self.actor_lora_param_dim
        elif self._use_head_residual():
            actor_hyper_output_dim = self.actor_head_param_dim
        else:
            actor_hyper_output_dim = self.actor_param_dim
        self.actor_hypernet = None
        if actor_hyper_output_dim is not None:
            self.actor_hypernet = build_hypernetwork(
                self.hypernet_type,
                self.meta_dim,
                hyper_hidden,
                actor_hyper_output_dim,
                dropout=hyper_dropout,
                target_layout=(
                    None
                    if self.hyper_adapter_mode == 'film' or self._use_compressed_residual()
                    else self.actor_layout
                ),
                head_mode=(
                    'flat'
                    if self.hyper_adapter_mode == 'film' or self._use_compressed_residual()
                    else self.hyper_head_mode
                ),
                use_bias=self.hyper_use_bias,
                head_init_gain=float(
                    cfg.get('hyper_actor_head_init_gain', self.hyper_head_init_gain)
                ),
                chunk_size=self.hyper_chunk_size,
                chunk_embed_dim=self.hyper_chunk_embed_dim,
                **self.actor_rf_init_config,
            ).to(self.device)
        if self.hyper_adapter_mode == 'film' and self.hyper_film_init_zero:
            self._zero_last_linear(self.actor_hypernet)

        self.graph_edge_index = None
        self.graph_edge_weight = None
        self.graph_critic = None
        self.base_value = None
        self.value_hypernet = None
        self.value_dims = []
        self.value_layout = []
        self.value_param_dim = 0
        self.value_head_layout = None
        self.value_head_param_dim = 0
        self.value_lora_layout = []
        self.value_lora_param_dim = 0
        self.value_film_param_dim = 0
        self.base_value_trainable = False
        if self.graph_critic_enabled:
            self.graph_edge_index, self.graph_edge_weight = self._build_directed_graph_edges()
            self.graph_critic = DirectedGraphCritic(
                self.policy_input_dim,
                hidden_dim=self.graph_critic_hidden_dim,
                num_layers=self.graph_critic_layers,
                num_heads=self.graph_critic_heads,
                dropout=self.graph_critic_dropout,
                use_edge_weight=self.graph_critic_use_edge_weight,
                edge_weight_scale=self.graph_critic_edge_weight_scale,
                global_pool=self.graph_critic_global_pool,
                meta_dim=self.meta_dim if self.graph_critic_film else None,
                film_hidden_dims=self.graph_critic_film_hidden,
                film_scale=self.graph_critic_film_scale,
                film_zero_init=True,
            ).to(self.device)
        else:
            self.value_dims = [self.value_input_dim] + self.value_hidden + [1]
            self.value_layout = self._build_layout_from_dims(self.value_dims)
            self.value_param_dim = self.value_layout[-1][-1]
            self.base_value = self._build_mlp(
                self.value_input_dim,
                self.value_hidden,
                1,
                final_activation=False,
            ).to(self.device)
            self.base_value_trainable = bool(
                self.hyper_residual or self.hyper_critic_adapter_mode == 'film'
            )
            for param in self.base_value.parameters():
                param.requires_grad = self.base_value_trainable
            self.value_head_layout, self.value_head_param_dim = self._build_head_layout(self.value_layout)
            self.value_lora_layout, self.value_lora_param_dim = self._build_lora_layout(
                self.value_layout,
                self.hyper_lora_value_rank,
                use_bias=self.hyper_lora_bias,
            )
            self.value_film_param_dim = 2 * sum(self.value_hidden)
            if self.hyper_critic_adapter_mode == 'film':
                value_hyper_output_dim = self.value_film_param_dim
            elif self._use_lora_residual():
                value_hyper_output_dim = self.value_lora_param_dim
            elif self._use_head_residual():
                value_hyper_output_dim = self.value_head_param_dim
            else:
                value_hyper_output_dim = self.value_param_dim
            self.value_hypernet = build_hypernetwork(
                self.value_hypernet_type,
                self.meta_dim,
                value_hyper_hidden,
                value_hyper_output_dim,
                dropout=hyper_dropout,
                target_layout=(
                    None
                    if self.hyper_critic_adapter_mode == 'film'
                    or self._use_compressed_residual()
                    else self.value_layout
                ),
                head_mode=(
                    'flat'
                    if self.hyper_critic_adapter_mode == 'film'
                    or self._use_compressed_residual()
                    else self.hyper_head_mode
                ),
                use_bias=self.hyper_use_bias,
                head_init_gain=float(cfg.get('hyper_value_head_init_gain', self.hyper_head_init_gain)),
                chunk_size=self.hyper_chunk_size,
                chunk_embed_dim=self.hyper_chunk_embed_dim,
                **self.value_rf_init_config,
            ).to(self.device)
            if (
                self.hyper_critic_adapter_mode == 'film'
                and self.hyper_film_init_zero
            ):
                self._zero_last_linear(self.value_hypernet)

        optimizer_params = self._optimizer_parameters()
        self.optimizer = optim.Adam(
            optimizer_params,
            lr=float(cfg.get('learning_rate', 3e-4)),
            eps=float(cfg.get('adam_eps', 1e-5)),
        )

        buffer_size = int(trainer_cfg.get('buffer_size', max(self.ppo_rollout_steps, 1)))
        self.rollout_buffer = deque(maxlen=buffer_size)
        self.replay_buffer = self.rollout_buffer
        self._transitions_since_update = 0
        self._cached_action_prob = None
        self._cached_value = None
        self._cached_cos_ids = None
        self._cached_cos_log_prob = None
        self._last_cos_diagnostics = {}
        self._last_residual_diagnostics = {}
        self._residual_episode_diagnostics = []
        self._last_abs_pressure = None

    def __repr__(self):
        critic_type = (
            f'centralized/{self.centralized_critic_mode}'
            if self.centralized_critic
            else 'local'
        )
        value_arch = 'directed_graph' if self.graph_critic_enabled else self.value_hypernet_type
        return (
            f"HyperLightPPOAgent(sub_agents={self.sub_agents}, state_dim={self.state_dim}, "
            f"policy_input_dim={self.policy_input_dim}, action_dim={self.action_space.n}, "
            f"actor={self.actor_arch}, "
            f"actor_hypernet={self.hypernet_type}/{self.hyper_adapter_mode}, "
            f"value_arch={value_arch}/{self.hyper_critic_adapter_mode}, "
            f"hyper_heads={self.hyper_head_mode}, "
            f"chunk={self.hyper_chunk_size}/{self.hyper_chunk_embed_dim}, "
            f"objective={self.policy_objective}, "
            f"embedding={self.embedding_mode}, topology={self.topology_aware_embedding}, "
            f"movement_encoder={self.movement_encoder_enabled}, "
            f"activation={self.activation}, "
            f"rf_init={self.actor_rf_init_config['rf_init']}, "
            f"residual={self.hyper_residual}/{self.hyper_residual_mode}@"
            f"{self.hyper_residual_actor_scale:g}/{self.hyper_residual_value_scale:g}, "
            f"lora_rank={self.hyper_lora_actor_rank}/{self.hyper_lora_value_rank}, "
            f"reward={self.reward_mode}, "
            f"cos={self.cos_enabled}@k={self.cos_top_k}, "
            f"value_chunk_size={self.value_chunk_size}, "
            f"test_action={self.test_action_mode}@T={self.test_temperature:g}, "
            f"critic={critic_type}, device={self.device})"
        )

    def _uses_pressure_reward(self):
        return self.reward_mode in ('pressure_abs', 'queue_pressure')

    def _build_generators(self):
        self.ob_generator = []
        self.reward_generator = []
        self.pressure_lanes = []
        pressure_norms = []
        self.phase_generator = []
        self.queue_generator = []
        self.delay_generator = []

        if self._uses_pressure_reward():
            self.world.subscribe(['lane_count'])

        max_ob_length = 0
        for inter in self.world.intersections:
            ob_gen = LaneVehicleGenerator(
                self.world,
                inter,
                self.state_features,
                in_only=True,
                average=None,
            )
            max_ob_length = max(max_ob_length, ob_gen.ob_length)
            self.ob_generator.append(ob_gen)
            self.reward_generator.append(
                LaneVehicleGenerator(
                    self.world,
                    inter,
                    ['lane_waiting_count'],
                    in_only=True,
                    average='all',
                    negative=True,
                )
            )
            in_lanes, out_lanes = self._build_pressure_lanes(inter)
            self.pressure_lanes.append((in_lanes, out_lanes))
            pressure_norms.append(float(max(len(in_lanes), 1)))
            self.phase_generator.append(
                IntersectionPhaseGenerator(
                    self.world,
                    inter,
                    ['phase'],
                    targets=['cur_phase'],
                    negative=False,
                )
            )
            self.queue_generator.append(
                LaneVehicleGenerator(
                    self.world,
                    inter,
                    ['lane_waiting_count'],
                    in_only=True,
                    average=None,
                    negative=False,
                )
            )
            self.delay_generator.append(
                LaneVehicleGenerator(
                    self.world,
                    inter,
                    ['lane_delay'],
                    in_only=True,
                    average='all',
                    negative=False,
                )
            )
        self.ob_length = int(max_ob_length)
        self.pressure_norms = np.asarray(pressure_norms, dtype=np.float32)

    def _lanes_for_road(self, inter, road):
        if hasattr(inter, 'road_lane_mapping') and road in inter.road_lane_mapping:
            return list(inter.road_lane_mapping[road])

        if isinstance(road, dict):
            road_id = road.get('id')
            lane_count = len(road.get('lanes', []))
            return [f'{road_id}_{idx}' for idx in range(lane_count)]

        return []

    def _build_pressure_lanes(self, inter):
        in_lanes = []
        out_lanes = []
        for road in getattr(inter, 'in_roads', []) or []:
            in_lanes.extend(self._lanes_for_road(inter, road))
        for road in getattr(inter, 'out_roads', []) or []:
            out_lanes.extend(self._lanes_for_road(inter, road))
        return in_lanes, out_lanes

    @staticmethod
    def _lane_count_value(lane_count, lane_id):
        if isinstance(lane_count, dict):
            return float(lane_count.get(lane_id, 0.0))
        return 0.0

    def _build_action_mask(self):
        mask = torch.zeros((self.sub_agents, self.action_space.n), dtype=torch.bool)
        for idx, phase_num in enumerate(self.phase_lengths):
            mask[idx, : max(1, int(phase_num))] = True
        return mask

    def _build_movement_token_spec(self):
        """Build padded lane-link tokens while preserving the generator's lane order."""
        raw_intersections = {
            item.get('id'): item
            for item in getattr(self.world, 'roadnet', {}).get('intersections', [])
        }
        node_specs = []
        for inter, ob_gen in zip(self.world.intersections, self.ob_generator):
            lane_ids = [lane for road_lanes in ob_gen.lanes for lane in road_lanes]
            if not lane_ids:
                raise ValueError(
                    f'movement encoder requires incoming lanes at intersection {getattr(inter, "id", "?")}'
                )
            lane_to_index = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}

            movements = []
            seen = set()
            for raw_movement in getattr(inter, 'lanelinks', []) or []:
                if raw_movement and isinstance(raw_movement[0], (tuple, list)):
                    candidates = [
                        (connection[0], connection[1])
                        for connection in raw_movement
                        if len(connection) >= 2
                    ]
                elif len(raw_movement) >= 2:
                    candidates = [(raw_movement[0], raw_movement[1])]
                else:
                    continue
                for movement in candidates:
                    if movement[0] not in lane_to_index or movement in seen:
                        continue
                    seen.add(movement)
                    movements.append(movement)
            if not movements:
                # SUMO and synthetic worlds may expose only controlled start lanes.
                movements = [(lane_id, None) for lane_id in lane_ids]

            phase_links = [
                set(tuple(link) for link in links)
                for links in (getattr(inter, 'phase_available_lanelinks', []) or [])
            ]
            phase_start_lanes = [
                set(lanes)
                for lanes in (getattr(inter, 'phase_available_startlanes', []) or [])
            ]
            turn_lookup = {}
            raw_inter = raw_intersections.get(getattr(inter, 'id', None), {})
            for road_link in raw_inter.get('roadLinks', []) or []:
                turn_name = str(road_link.get('type', '')).lower()
                if 'left' in turn_name:
                    turn_idx = 0
                elif 'straight' in turn_name:
                    turn_idx = 1
                elif 'right' in turn_name:
                    turn_idx = 2
                else:
                    turn_idx = 3
                start_road = road_link.get('startRoad')
                end_road = road_link.get('endRoad')
                for lane_link in road_link.get('laneLinks', []) or []:
                    movement = (
                        f"{start_road}_{lane_link.get('startLaneIndex')}",
                        f"{end_road}_{lane_link.get('endLaneIndex')}",
                    )
                    turn_lookup[movement] = turn_idx
            node_specs.append(
                {
                    'lane_ids': lane_ids,
                    'lane_to_index': lane_to_index,
                    'movements': movements,
                    'phase_links': phase_links,
                    'phase_start_lanes': phase_start_lanes,
                    'turn_lookup': turn_lookup,
                }
            )

        max_movements = max(len(spec['movements']) for spec in node_specs)
        feature_count = len(self.state_features)
        feature_indices = np.zeros(
            (self.sub_agents, max_movements, feature_count),
            dtype=np.int64,
        )
        token_mask = np.zeros((self.sub_agents, max_movements), dtype=np.bool_)
        phase_availability = np.zeros(
            (self.sub_agents, max_movements, self.action_space.n),
            dtype=np.float32,
        )
        source_position = np.zeros((self.sub_agents, max_movements), dtype=np.float32)
        movement_position = np.zeros((self.sub_agents, max_movements), dtype=np.float32)
        turn_features = np.zeros((self.sub_agents, max_movements, 4), dtype=np.float32)

        for node_idx, spec in enumerate(node_specs):
            lane_count = len(spec['lane_ids'])
            destination_indices = []
            for movement_idx, (_, destination_lane) in enumerate(spec['movements']):
                try:
                    destination_indices.append(int(str(destination_lane).rsplit('_', 1)[-1]))
                except (TypeError, ValueError):
                    destination_indices.append(movement_idx)
            max_destination_idx = max(destination_indices) if destination_indices else 0
            for movement_idx, movement in enumerate(spec['movements']):
                source_lane, _ = movement
                source_idx = spec['lane_to_index'][source_lane]
                for feature_idx in range(feature_count):
                    flat_idx = feature_idx * lane_count + source_idx
                    if flat_idx >= self.ob_length:
                        raise ValueError('movement feature index exceeds padded observation length')
                    feature_indices[node_idx, movement_idx, feature_idx] = flat_idx
                token_mask[node_idx, movement_idx] = True
                source_position[node_idx, movement_idx] = source_idx / float(max(1, lane_count - 1))
                movement_position[node_idx, movement_idx] = destination_indices[movement_idx] / float(
                    max(1, max_destination_idx)
                )
                turn_idx = int(spec['turn_lookup'].get(movement, 3))
                turn_features[node_idx, movement_idx, turn_idx] = 1.0

                for action_idx in range(min(int(self.phase_lengths[node_idx]), self.action_space.n)):
                    if action_idx < len(spec['phase_links']) and movement[1] is not None:
                        available = movement in spec['phase_links'][action_idx]
                    elif action_idx < len(spec['phase_start_lanes']):
                        available = source_lane in spec['phase_start_lanes'][action_idx]
                    else:
                        available = False
                    phase_availability[node_idx, movement_idx, action_idx] = float(available)

        self.movement_feature_indices = torch.tensor(
            feature_indices,
            dtype=torch.long,
            device=self.device,
        )
        self.movement_token_mask = torch.tensor(
            token_mask,
            dtype=torch.bool,
            device=self.device,
        )
        self.movement_phase_availability = torch.tensor(
            phase_availability,
            dtype=torch.float32,
            device=self.device,
        )
        self.movement_source_position = torch.tensor(
            source_position,
            dtype=torch.float32,
            device=self.device,
        )
        self.movement_position = torch.tensor(
            movement_position,
            dtype=torch.float32,
            device=self.device,
        )
        self.movement_turn_features = torch.tensor(
            turn_features,
            dtype=torch.float32,
            device=self.device,
        )

    def _encode_policy_state(self, state_tensor):
        if self.movement_encoder is None:
            return state_tensor

        batch_size = state_tensor.shape[0]
        feature_indices = self.movement_feature_indices.reshape(self.sub_agents, -1)
        gather_index = feature_indices.unsqueeze(0).expand(batch_size, -1, -1)
        dynamic_features = torch.gather(
            state_tensor[..., :self.ob_length],
            dim=2,
            index=gather_index,
        ).view(
            batch_size,
            self.sub_agents,
            self.movement_feature_indices.shape[1],
            len(self.state_features),
        )
        dynamic_features = dynamic_features.masked_fill(
            ~self.movement_token_mask.unsqueeze(0).unsqueeze(-1),
            0.0,
        )

        if not self.phase:
            current_phase = state_tensor.new_zeros(
                batch_size,
                self.sub_agents,
                self.action_space.n,
            )
        elif self.one_hot:
            current_phase = state_tensor[
                ...,
                self.ob_length:self.ob_length + self.action_space.n,
            ]
        else:
            phase_index = state_tensor[..., self.ob_length].round().long()
            phase_index = phase_index.clamp(0, self.action_space.n - 1)
            current_phase = F.one_hot(
                phase_index,
                num_classes=self.action_space.n,
            ).to(dtype=state_tensor.dtype)

        return self.movement_encoder(
            dynamic_features,
            self.movement_token_mask,
            self.movement_phase_availability,
            current_phase,
            source_position=self.movement_source_position,
            movement_position=self.movement_position,
            static_features=self.movement_turn_features,
        )

    def _build_directed_graph_edges(self):
        edge_index = None
        edge_weight = None
        if hasattr(self.world, 'get_adjacency'):
            try:
                edge_index, edge_weight = self.world.get_adjacency()
            except Exception:
                edge_index, edge_weight = None, None

        if edge_index is None:
            try:
                graph = Registry.mapping['world_mapping']['graph_setting'].graph
                sparse_adj = np.asarray(graph.get('sparse_adj', []), dtype=np.int64)
                if sparse_adj.size > 0:
                    edge_index = torch.tensor(sparse_adj.T, dtype=torch.long)
                    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
            except (AttributeError, KeyError, TypeError, ValueError):
                edge_index, edge_weight = None, None

        if edge_index is None:
            edges = []
            weights = []
            id_to_idx = {
                inter_id: idx
                for idx, inter_id in enumerate(getattr(self.world, 'intersection_ids', []))
            }
            for road in getattr(self.world, 'roadnet', {}).get('roads', []):
                src = id_to_idx.get(road.get('startIntersection'))
                dst = id_to_idx.get(road.get('endIntersection'))
                if src is None or dst is None:
                    continue
                edges.append((src, dst))
                length = road.get('length')
                if length is None and hasattr(self.world, 'get_road_length'):
                    length = self.world.get_road_length(road)
                weights.append(1.0 / (float(length or 1.0) + 1e-3))
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_weight = torch.tensor(weights, dtype=torch.float32)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_weight = torch.empty((0,), dtype=torch.float32)

        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError('world adjacency must return edge_index with shape [2, E]')
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
        edge_weight = torch.as_tensor(edge_weight, dtype=torch.float32).reshape(-1)
        if edge_weight.numel() != edge_index.shape[1]:
            raise ValueError('world adjacency edge weights do not match edge count')

        valid = (
            (edge_index[0] >= 0)
            & (edge_index[0] < self.sub_agents)
            & (edge_index[1] >= 0)
            & (edge_index[1] < self.sub_agents)
        )
        edge_index = edge_index[:, valid]
        edge_weight = edge_weight[valid]

        direction = self.graph_message_direction
        if direction in ('downstream_to_upstream', 'reverse', 'spillback'):
            edge_index = edge_index.flip(0)
        elif direction in ('upstream_to_downstream', 'forward', 'road'):
            pass
        elif direction in ('bidirectional', 'both'):
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
        else:
            raise ValueError(f'Unknown graph_message_direction: {self.graph_message_direction}')
        return edge_index.to(self.device), edge_weight.to(self.device)

    @staticmethod
    def _zero_last_linear(module):
        linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
        if not linear_layers:
            raise TypeError('hypernetwork does not contain a Linear output layer')
        nn.init.zeros_(linear_layers[-1].weight)
        if linear_layers[-1].bias is not None:
            nn.init.zeros_(linear_layers[-1].bias)

    @staticmethod
    def _as_int_list(value):
        if value is None:
            return []
        if isinstance(value, (tuple, list)):
            return [int(item) for item in value]
        return [int(value)]

    @staticmethod
    def _build_mlp(input_dim, hidden_dims, output_dim, final_activation=False):
        dims = [int(input_dim)] + [int(item) for item in hidden_dims] + [int(output_dim)]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2 or final_activation:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _build_topology_features(self):
        features = []
        for idx, inter in enumerate(self.world.intersections):
            point = self._intersection_point(idx, inter)
            in_roads = getattr(inter, 'in_roads', []) or []
            out_roads = getattr(inter, 'out_roads', []) or []
            in_lane_count = float(sum(self._road_lane_count(road) for road in in_roads))
            out_lane_count = float(sum(self._road_lane_count(road) for road in out_roads))
            in_degree = float(len(in_roads))
            out_degree = float(len(out_roads))
            neighbor_count = float(len(self._neighbor_intersections(inter)))
            phase_count = float(max(1, len(getattr(inter, 'phases', []))))
            startlane_count = float(len(getattr(inter, 'startlanes', []) or []))
            features.append(
                [
                    float(point[0]),
                    float(point[1]),
                    in_lane_count,
                    out_lane_count,
                    in_degree,
                    out_degree,
                    in_degree + out_degree,
                    neighbor_count,
                    phase_count,
                    startlane_count,
                ]
            )

        feature_array = np.asarray(features, dtype=np.float32)
        feature_array = self._normalize_topology_features(feature_array)
        names = [
            'x',
            'y',
            'in_lane_count',
            'out_lane_count',
            'in_degree',
            'out_degree',
            'node_degree',
            'neighbor_count',
            'phase_count',
            'startlane_count',
        ]
        return feature_array, names

    def _intersection_point(self, idx, inter):
        if hasattr(self.world, 'intersection_points'):
            points = np.asarray(self.world.intersection_points, dtype=np.float32)
            if idx < len(points):
                return points[idx]

        inter_id = getattr(inter, 'id', None)
        for item in getattr(self.world, 'roadnet', {}).get('intersections', []):
            if item.get('id') == inter_id and 'point' in item:
                point = item['point']
                return np.asarray([point.get('x', 0.0), point.get('y', 0.0)], dtype=np.float32)
        return np.zeros((2,), dtype=np.float32)

    @staticmethod
    def _road_lane_count(road):
        if isinstance(road, dict):
            return len(road.get('lanes', []) or [])
        return 0

    def _neighbor_intersections(self, inter):
        neighbors = set()
        inter_id = getattr(inter, 'id', None)
        for road in (getattr(inter, 'in_roads', []) or []):
            if isinstance(road, dict):
                start = road.get('startIntersection')
                if start in getattr(self.world, 'intersection_ids', []) and start != inter_id:
                    neighbors.add(start)
        for road in (getattr(inter, 'out_roads', []) or []):
            if isinstance(road, dict):
                end = road.get('endIntersection')
                if end in getattr(self.world, 'intersection_ids', []) and end != inter_id:
                    neighbors.add(end)
        return neighbors

    @staticmethod
    def _normalize_topology_features(features):
        if features.size == 0:
            return features
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (features - mean) / std

    def _build_cos_pairwise_metrics(self):
        hop_matrix = np.full((self.sub_agents, self.sub_agents), np.inf, dtype=np.float32)
        np.fill_diagonal(hop_matrix, 0.0)

        id_to_idx = {
            inter_id: idx
            for idx, inter_id in enumerate(getattr(self.world, 'intersection_ids', []))
        }
        for idx, inter in enumerate(self.world.intersections):
            for neighbor_id in self._neighbor_intersections(inter):
                neighbor_idx = id_to_idx.get(neighbor_id)
                if neighbor_idx is None or neighbor_idx >= self.sub_agents:
                    continue
                hop_matrix[idx, neighbor_idx] = 1.0
                hop_matrix[neighbor_idx, idx] = 1.0

        for pivot in range(self.sub_agents):
            hop_matrix = np.minimum(
                hop_matrix,
                hop_matrix[:, pivot:pivot + 1] + hop_matrix[pivot:pivot + 1, :],
            )
        finite_mask = np.isfinite(hop_matrix)
        if not finite_mask.all():
            max_finite = float(hop_matrix[finite_mask].max()) if finite_mask.any() else 0.0
            hop_matrix = np.where(
                finite_mask,
                hop_matrix,
                max(max_finite + 1.0, float(self.sub_agents)),
            )

        points = np.asarray(
            [
                self._intersection_point(idx, inter)
                for idx, inter in enumerate(self.world.intersections)
            ],
            dtype=np.float32,
        )
        deltas = points[:, None, :] - points[None, :, :]
        distance_matrix = np.sqrt(np.maximum(np.sum(deltas * deltas, axis=-1), 0.0)).astype(np.float32)

        return (
            torch.tensor(hop_matrix, dtype=torch.float32, device=self.device),
            torch.tensor(distance_matrix, dtype=torch.float32, device=self.device),
        )

    def _build_layout_from_module(self, module):
        layout = []
        offset = 0
        for name, param in module.named_parameters():
            numel = int(param.numel())
            layout.append((name, tuple(param.shape), offset, offset + numel))
            offset += numel
        return layout

    @staticmethod
    def _build_linear_layout_from_module(module):
        """Layout restricted to nn.Linear submodules, in registration order.

        Used for hypernetwork-generated IRU actors: LayerNorm affine params
        (1-D, no (out_dim, in_dim) matrix shape) can't be sliced by the
        layerwise/chunked heads, so only Linear weight/bias entries are
        included here. LayerNorm stays a real, shared, trainable submodule.
        """
        layout = []
        offset = 0
        for mod_name, submodule in module.named_modules():
            if not isinstance(submodule, nn.Linear):
                continue
            prefix = f'{mod_name}.' if mod_name else ''
            weight = submodule.weight
            layout.append((f'{prefix}weight', tuple(weight.shape), offset, offset + weight.numel()))
            offset += weight.numel()
            if submodule.bias is not None:
                bias = submodule.bias
                layout.append((f'{prefix}bias', tuple(bias.shape), offset, offset + bias.numel()))
                offset += bias.numel()
        return layout

    @staticmethod
    def _build_layout_from_dims(dims):
        layout = []
        offset = 0
        for layer_idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            weight_numel = out_dim * in_dim
            bias_numel = out_dim
            layout.append(
                (
                    f'layer{layer_idx}',
                    (out_dim, in_dim),
                    offset,
                    offset + weight_numel,
                    offset + weight_numel + bias_numel,
                )
            )
            offset += weight_numel + bias_numel
        return layout

    def _use_lora_residual(self):
        return self.hyper_residual and self.hyper_residual_mode == 'lora'

    def _use_head_residual(self):
        return self.hyper_residual and self.hyper_residual_mode == 'head'

    def _use_compressed_residual(self):
        return self._use_lora_residual() or self._use_head_residual()

    @staticmethod
    def _linear_layout_entries(layout):
        if not layout:
            return []

        first = layout[0]
        entries = []
        if len(first) == 4:
            if len(layout) % 2 != 0:
                raise ValueError('Expected weight/bias pairs in named target layout')
            for idx in range(0, len(layout), 2):
                weight_name, weight_shape, weight_start, weight_end = layout[idx]
                bias_name, bias_shape, bias_start, bias_end = layout[idx + 1]
                if not str(weight_name).endswith('.weight') or not str(bias_name).endswith('.bias'):
                    raise ValueError('Expected target layout to alternate weight then bias')
                out_dim, in_dim = tuple(weight_shape)
                entries.append(
                    {
                        'name': str(weight_name).rsplit('.', 1)[0],
                        'shape': (int(out_dim), int(in_dim)),
                        'weight_start': int(weight_start),
                        'weight_end': int(weight_end),
                        'bias_start': int(bias_start),
                        'bias_end': int(bias_end),
                    }
                )
            return entries

        for item in layout:
            if len(item) != 5:
                raise ValueError('Expected target layout entries with 4 or 5 fields')
            name, shape, weight_start, bias_start, bias_end = item
            out_dim, in_dim = tuple(shape)
            entries.append(
                {
                    'name': str(name),
                    'shape': (int(out_dim), int(in_dim)),
                    'weight_start': int(weight_start),
                    'weight_end': int(bias_start),
                    'bias_start': int(bias_start),
                    'bias_end': int(bias_end),
                }
            )
        return entries

    def _build_lora_layout(self, target_layout, rank, use_bias=False):
        layout = []
        offset = 0
        for entry in self._linear_layout_entries(target_layout):
            out_dim, in_dim = entry['shape']
            layer_rank = max(1, min(int(rank), int(out_dim), int(in_dim)))
            a_dim = layer_rank * int(in_dim)
            b_dim = int(out_dim) * layer_rank
            bias_dim = int(out_dim) if use_bias else 0
            lora_entry = dict(entry)
            lora_entry.update(
                {
                    'rank': layer_rank,
                    'a_start': offset,
                    'a_end': offset + a_dim,
                    'b_start': offset + a_dim,
                    'b_end': offset + a_dim + b_dim,
                    'lora_bias_start': offset + a_dim + b_dim,
                    'lora_bias_end': offset + a_dim + b_dim + bias_dim,
                }
            )
            layout.append(lora_entry)
            offset += a_dim + b_dim + bias_dim
        return layout, offset

    def _build_head_layout(self, target_layout):
        entries = self._linear_layout_entries(target_layout)
        if not entries:
            raise ValueError('head-only residual requires at least one linear target layer')

        entry = dict(entries[-1])
        weight_dim = int(entry['weight_end']) - int(entry['weight_start'])
        bias_dim = int(entry['bias_end']) - int(entry['bias_start'])
        entry.update(
            {
                'head_weight_start': 0,
                'head_weight_end': weight_dim,
                'head_bias_start': weight_dim,
                'head_bias_end': weight_dim + bias_dim,
            }
        )
        return entry, weight_dim + bias_dim

    def _activate(self, x):
        if self.activation == 'tanh':
            return torch.tanh(x)
        return F.relu(x)

    @staticmethod
    def _flat_module_parameters(module):
        return torch.cat([param.reshape(-1) for param in module.parameters()], dim=0)

    def _compose_residual_theta(self, generated_theta, base_module, scale):
        if not self.hyper_residual:
            return generated_theta, None, None

        base_theta = self._flat_module_parameters(base_module).to(generated_theta.device)
        if base_theta.shape[-1] != generated_theta.shape[-1]:
            raise RuntimeError(
                f"Residual base/generator parameter mismatch: "
                f"base={base_theta.shape[-1]}, generated={generated_theta.shape[-1]}"
            )
        view_shape = (1,) * (generated_theta.dim() - 1) + (base_theta.shape[-1],)
        base_theta = base_theta.view(view_shape)
        delta_theta = generated_theta * float(scale)
        return base_theta + delta_theta, base_theta, delta_theta

    def _compose_lora_theta(self, lora_theta, base_module, lora_layout, scale):
        base_theta = self._flat_module_parameters(base_module).to(lora_theta.device)
        view_shape = (1,) * (lora_theta.dim() - 1) + (base_theta.shape[-1],)
        base_theta = base_theta.view(view_shape)

        delta_chunks = []
        for entry in lora_layout:
            out_dim, in_dim = entry['shape']
            rank = int(entry['rank'])
            a = lora_theta[..., entry['a_start']:entry['a_end']].view(
                *lora_theta.shape[:-1],
                rank,
                in_dim,
            )
            b = lora_theta[..., entry['b_start']:entry['b_end']].view(
                *lora_theta.shape[:-1],
                out_dim,
                rank,
            )
            delta_weight = torch.matmul(b, a).reshape(*lora_theta.shape[:-1], out_dim * in_dim)
            delta_weight = delta_weight * (float(scale) / float(rank))
            delta_chunks.append(delta_weight)

            if entry['lora_bias_end'] > entry['lora_bias_start']:
                delta_bias = lora_theta[..., entry['lora_bias_start']:entry['lora_bias_end']]
                delta_bias = delta_bias * float(scale)
            else:
                delta_bias = lora_theta.new_zeros(*lora_theta.shape[:-1], out_dim)
            delta_chunks.append(delta_bias)

        delta_theta = torch.cat(delta_chunks, dim=-1)
        if delta_theta.shape[-1] != base_theta.shape[-1]:
            raise RuntimeError(
                f"LoRA residual base/delta mismatch: "
                f"base={base_theta.shape[-1]}, delta={delta_theta.shape[-1]}"
            )
        return base_theta + delta_theta, base_theta, delta_theta

    def _compose_head_theta(self, head_theta, base_module, head_layout, scale):
        """Materialize head-only residuals for legacy/equivalence use.

        The policy/value forward paths intentionally bypass this helper so a
        compact generated head does not allocate a full per-agent parameter
        vector.  Keeping it available is useful for checkpoint validation and
        numerical equivalence tests.
        """
        base_theta = self._flat_module_parameters(base_module).to(head_theta.device)
        view_shape = (1,) * (head_theta.dim() - 1) + (base_theta.shape[-1],)
        base_theta = base_theta.view(view_shape)

        expected_dim = int(head_layout['head_bias_end'])
        if head_theta.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Head residual generator mismatch: "
                f"expected={expected_dim}, generated={head_theta.shape[-1]}"
            )

        delta_theta = head_theta.new_zeros(*head_theta.shape[:-1], base_theta.shape[-1])
        weight_delta = head_theta[
            ...,
            head_layout['head_weight_start']:head_layout['head_weight_end'],
        ] * float(scale)
        bias_delta = head_theta[
            ...,
            head_layout['head_bias_start']:head_layout['head_bias_end'],
        ] * float(scale)
        delta_theta[
            ...,
            head_layout['weight_start']:head_layout['weight_end'],
        ] = weight_delta
        delta_theta[
            ...,
            head_layout['bias_start']:head_layout['bias_end'],
        ] = bias_delta
        return base_theta + delta_theta, base_theta, delta_theta

    def _actor_theta_from_meta(self, meta):
        if self.hyper_adapter_mode in ('film', 'none'):
            raise RuntimeError(
                f'{self.hyper_adapter_mode} actor does not materialize dense generated weights'
            )
        generated_theta = self.actor_hypernet(meta)
        if self._use_lora_residual():
            return self._compose_lora_theta(
                generated_theta,
                self.base_actor,
                self.actor_lora_layout,
                self.hyper_residual_actor_scale,
            )
        if self._use_head_residual():
            return self._compose_head_theta(
                generated_theta,
                self.base_actor,
                self.actor_head_layout,
                self.hyper_residual_actor_scale,
            )
        return self._compose_residual_theta(
            generated_theta,
            self.base_actor,
            self.hyper_residual_actor_scale,
        )

    def _value_theta_from_meta(self, meta):
        if self.hyper_critic_adapter_mode == 'film':
            raise RuntimeError('FiLM critic does not materialize dense generated weights')
        generated_theta = self.value_hypernet(meta)
        if self._use_lora_residual():
            return self._compose_lora_theta(
                generated_theta,
                self.base_value,
                self.value_lora_layout,
                self.hyper_residual_value_scale,
            )
        if self._use_head_residual():
            return self._compose_head_theta(
                generated_theta,
                self.base_value,
                self.value_head_layout,
                self.hyper_residual_value_scale,
            )
        return self._compose_residual_theta(
            generated_theta,
            self.base_value,
            self.hyper_residual_value_scale,
        )

    @staticmethod
    def _theta_diagnostics(prefix, base_theta, delta_theta, theta):
        if base_theta is None or delta_theta is None:
            return {}

        with torch.no_grad():
            flat_delta = delta_theta.reshape(-1, delta_theta.shape[-1])
            flat_theta = theta.reshape(-1, theta.shape[-1])
            base_flat = base_theta.reshape(-1)
            base_norm = base_flat.norm(p=2).clamp_min(1e-8)
            delta_norm = flat_delta.norm(p=2, dim=-1)
            theta_norm = flat_theta.norm(p=2, dim=-1)
            return {
                f'{prefix}_base_norm': float(base_norm.detach().cpu().item()),
                f'{prefix}_delta_norm': float(delta_norm.mean().detach().cpu().item()),
                f'{prefix}_delta_base_ratio': float((delta_norm.mean() / base_norm).detach().cpu().item()),
                f'{prefix}_delta_max_abs': float(delta_theta.abs().max().detach().cpu().item()),
                f'{prefix}_theta_norm': float(theta_norm.mean().detach().cpu().item()),
            }

    @staticmethod
    def _theta_slice_diagnostics(prefix, base_theta, delta_theta, theta, start, end):
        if base_theta is None or delta_theta is None:
            return {}

        with torch.no_grad():
            base_slice = base_theta[..., start:end].reshape(-1, end - start)
            flat_delta = delta_theta[..., start:end].reshape(-1, end - start)
            flat_theta = theta[..., start:end].reshape(-1, end - start)
            base_norm = base_slice.norm(p=2, dim=-1).mean().clamp_min(1e-8)
            delta_norm = flat_delta.norm(p=2, dim=-1)
            theta_norm = flat_theta.norm(p=2, dim=-1)
            return {
                f'{prefix}_base_norm': float(base_norm.detach().cpu().item()),
                f'{prefix}_delta_norm': float(delta_norm.mean().detach().cpu().item()),
                f'{prefix}_delta_base_ratio': float((delta_norm.mean() / base_norm).detach().cpu().item()),
                f'{prefix}_delta_max_abs': float(delta_theta[..., start:end].abs().max().detach().cpu().item()),
                f'{prefix}_theta_norm': float(theta_norm.mean().detach().cpu().item()),
            }

    def _head_diagnostics(self, prefix, base_theta, delta_theta, theta, head_layout):
        return self._theta_slice_diagnostics(
            prefix,
            base_theta,
            delta_theta,
            theta,
            int(head_layout['weight_start']),
            int(head_layout['bias_end']),
        )

    @staticmethod
    def _last_linear(module):
        linear_layers = [child for child in module.modules() if isinstance(child, nn.Linear)]
        if not linear_layers:
            raise RuntimeError('head-only residual requires a base module with a linear head')
        return linear_layers[-1]

    def _scaled_head_delta(self, head_theta, head_layout, scale):
        expected_dim = int(head_layout['head_bias_end'])
        if head_theta.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Head residual generator mismatch: "
                f"expected={expected_dim}, generated={head_theta.shape[-1]}"
            )
        return head_theta * float(scale)

    def _head_linear_forward(self, hidden, scaled_head_delta, base_module, head_layout):
        """Apply a shared linear head plus a compact per-agent residual."""
        base_head = self._last_linear(base_module)
        out_dim, in_dim = head_layout['shape']
        if base_head.in_features != in_dim or base_head.out_features != out_dim:
            raise RuntimeError(
                'Head residual/base head shape mismatch: '
                f'layout={(out_dim, in_dim)}, '
                f'base={(base_head.out_features, base_head.in_features)}'
            )

        delta_weight = scaled_head_delta[
            ...,
            head_layout['head_weight_start']:head_layout['head_weight_end'],
        ].view(*scaled_head_delta.shape[:-1], out_dim, in_dim)
        delta_bias = scaled_head_delta[
            ...,
            head_layout['head_bias_start']:head_layout['head_bias_end'],
        ].view(*scaled_head_delta.shape[:-1], out_dim)
        base_output = base_head(hidden)
        delta_output = torch.einsum('bni,bnoi->bno', hidden, delta_weight) + delta_bias
        return base_output + delta_output

    def _actor_head_residual_forward(self, policy_state, head_theta):
        if self.actor_arch == 'iru':
            hidden = self.base_actor.forward_features(policy_state)
        else:
            hidden = self._activate(self.base_actor.fc1(policy_state))
            hidden = self._activate(self.base_actor.fc2(hidden))
        scaled_head_delta = self._scaled_head_delta(
            head_theta,
            self.actor_head_layout,
            self.hyper_residual_actor_scale,
        )
        logits = self._head_linear_forward(
            hidden,
            scaled_head_delta,
            self.base_actor,
            self.actor_head_layout,
        )
        return logits, scaled_head_delta

    def _value_base_features(self, value_input):
        linear_layers = [
            layer for layer in self.base_value.children() if isinstance(layer, nn.Linear)
        ]
        if not linear_layers:
            raise RuntimeError('head-only residual expects the base value head to be linear')
        hidden = value_input
        for layer in linear_layers[:-1]:
            hidden = self._activate(layer(hidden))
        return hidden

    def _value_film_forward(self, value_input, film_params):
        if film_params.shape[-1] != self.value_film_param_dim:
            raise RuntimeError(
                'Critic FiLM generator mismatch: '
                f'expected={self.value_film_param_dim}, actual={film_params.shape[-1]}'
            )
        if film_params.shape[:-1] != value_input.shape[:-1]:
            raise RuntimeError(
                'Critic FiLM leading dimensions must match value inputs: '
                f'film={tuple(film_params.shape[:-1])}, '
                f'value={tuple(value_input.shape[:-1])}'
            )

        linear_layers = [
            layer for layer in self.base_value.children() if isinstance(layer, nn.Linear)
        ]
        if len(linear_layers) != len(self.value_hidden) + 1:
            raise RuntimeError(
                'FiLM critic expects one linear layer per configured hidden layer '
                'plus the scalar value head'
            )

        offset = 0
        hidden = value_input
        for layer, hidden_dim in zip(linear_layers[:-1], self.value_hidden):
            hidden = self._activate(layer(hidden))
            gamma = film_params[..., offset:offset + hidden_dim]
            offset += hidden_dim
            beta = film_params[..., offset:offset + hidden_dim]
            offset += hidden_dim
            hidden = hidden * (
                1.0 + self.hyper_film_scale * torch.tanh(gamma)
            )
            hidden = hidden + self.hyper_film_scale * torch.tanh(beta)
        return linear_layers[-1](hidden).squeeze(-1)

    def _value_film_diagnostics(self, film_params):
        split_sizes = []
        for hidden_dim in self.value_hidden:
            split_sizes.extend((hidden_dim, hidden_dim))
        chunks = torch.split(film_params, split_sizes, dim=-1)
        gamma = torch.cat(chunks[0::2], dim=-1)
        beta = torch.cat(chunks[1::2], dim=-1)
        return {
            'critic_adapter_is_film': 1.0,
            'value_film_gamma_abs_mean': float(
                gamma.abs().mean().detach().cpu().item()
            ),
            'value_film_beta_abs_mean': float(
                beta.abs().mean().detach().cpu().item()
            ),
            'value_film_param_dim': float(self.value_film_param_dim),
        }

    def _value_head_residual_forward(self, value_input, head_theta):
        hidden = self._value_base_features(value_input)
        scaled_head_delta = self._scaled_head_delta(
            head_theta,
            self.value_head_layout,
            self.hyper_residual_value_scale,
        )
        values = self._head_linear_forward(
            hidden,
            scaled_head_delta,
            self.base_value,
            self.value_head_layout,
        )
        return values.squeeze(-1), scaled_head_delta

    def _compact_head_diagnostics(
        self,
        prefix,
        base_module,
        scaled_head_delta,
    ):
        """Match legacy full-theta diagnostics without materializing theta."""
        with torch.no_grad():
            base_head = self._last_linear(base_module)
            base_head_flat = torch.cat(
                [base_head.weight.reshape(-1), base_head.bias.reshape(-1)],
                dim=0,
            )
            base_sq_norm = sum(param.detach().square().sum() for param in base_module.parameters())
            base_norm = base_sq_norm.sqrt().clamp_min(1e-8)
            base_head_sq_norm = base_head_flat.square().sum()
            base_non_head_sq_norm = (base_sq_norm - base_head_sq_norm).clamp_min(0.0)

            flat_delta = scaled_head_delta.reshape(-1, scaled_head_delta.shape[-1])
            delta_norm = flat_delta.norm(p=2, dim=-1)
            head_theta = flat_delta + base_head_flat.view(1, -1)
            head_theta_norm = head_theta.norm(p=2, dim=-1)
            theta_norm = (base_non_head_sq_norm + head_theta.square().sum(dim=-1)).sqrt()
            base_head_norm = base_head_flat.norm(p=2).clamp_min(1e-8)
            delta_mean = delta_norm.mean()

            return {
                f'{prefix}_base_norm': float(base_norm.detach().cpu().item()),
                f'{prefix}_delta_norm': float(delta_mean.detach().cpu().item()),
                f'{prefix}_delta_base_ratio': float((delta_mean / base_norm).detach().cpu().item()),
                f'{prefix}_delta_max_abs': float(
                    scaled_head_delta.abs().max().detach().cpu().item()
                ),
                f'{prefix}_theta_norm': float(theta_norm.mean().detach().cpu().item()),
                f'{prefix}_head_base_norm': float(base_head_norm.detach().cpu().item()),
                f'{prefix}_head_delta_norm': float(delta_mean.detach().cpu().item()),
                f'{prefix}_head_delta_base_ratio': float(
                    (delta_mean / base_head_norm).detach().cpu().item()
                ),
                f'{prefix}_head_delta_max_abs': float(
                    scaled_head_delta.abs().max().detach().cpu().item()
                ),
                f'{prefix}_head_theta_norm': float(
                    head_theta_norm.mean().detach().cpu().item()
                ),
            }

    @staticmethod
    def _mean_diagnostics(diagnostics):
        if not diagnostics:
            return {}
        keys = sorted({key for item in diagnostics for key in item})
        averaged = {}
        for key in keys:
            values = [
                float(item[key])
                for item in diagnostics
                if key in item and np.isfinite(float(item[key]))
            ]
            if values:
                averaged[key] = float(np.mean(values))
        return averaged

    def _policy_diagnostics(self, meta, raw_logits, values):
        with torch.no_grad():
            mask = self.action_mask.unsqueeze(0).expand_as(raw_logits)
            valid_logits = raw_logits.masked_select(mask)
            flat_meta = meta.reshape(-1, meta.shape[-1])
            diagnostics = {
                'hyper_residual_actor_scale': float(self.hyper_residual_actor_scale),
                'hyper_residual_value_scale': float(self.hyper_residual_value_scale),
                'hyper_residual_is_lora': float(1.0 if self._use_lora_residual() else 0.0),
                'hyper_residual_is_head': float(1.0 if self._use_head_residual() else 0.0),
                'hyper_head_actor_param_dim': float(self.actor_head_param_dim),
                'hyper_head_value_param_dim': float(self.value_head_param_dim),
                'hyper_lora_actor_rank': float(self.hyper_lora_actor_rank),
                'hyper_lora_value_rank': float(self.hyper_lora_value_rank),
                'meta_norm': float(flat_meta.norm(p=2, dim=-1).mean().detach().cpu().item()),
                'meta_std': float(flat_meta.std(unbiased=False).detach().cpu().item()),
                'value_std': float(values.std(unbiased=False).detach().cpu().item()),
                'value_abs_mean': float(values.abs().mean().detach().cpu().item()),
            }
            if valid_logits.numel() > 0:
                diagnostics['policy_logit_std'] = float(
                    valid_logits.std(unbiased=False).detach().cpu().item()
                )
                diagnostics['policy_logit_abs_mean'] = float(
                    valid_logits.abs().mean().detach().cpu().item()
                )
            return diagnostics

    def _agent_meta(self, batch_size):
        meta = self.agent_embeddings
        if self.topology_encoder is not None:
            meta = meta + self.topology_encoder(self.registered_topology_features)
        return meta.unsqueeze(0).expand(batch_size, -1, -1)

    def _cos_parameters(self):
        modules = [
            self.cos_state_encoder,
            self.cos_meta_encoder,
            self.cos_selector,
            self.cos_team_projector,
        ]
        params = []
        for module in modules:
            if module is not None:
                params.extend(module.parameters())
        return params

    def _cos_representations(self, state_tensor, base_meta):
        parts = []
        if self.cos_state_encoder is not None:
            parts.append(self.cos_state_encoder(state_tensor))
        if self.cos_meta_encoder is not None:
            parts.append(self.cos_meta_encoder(base_meta))
        if not parts:
            raise RuntimeError('CoS is enabled but no feature encoder is configured')
        if len(parts) == 1:
            return parts[0]
        return torch.stack(parts, dim=0).sum(dim=0)

    def _cos_logits(self, cos_repr):
        logits = self.cos_selector(cos_repr)
        if self.cos_self_bias != 0.0:
            eye = torch.eye(self.sub_agents, dtype=logits.dtype, device=logits.device)
            logits = logits + self.cos_self_bias * eye.unsqueeze(0)
        if self.cos_logit_clip > 0.0:
            logits = logits.clamp(-self.cos_logit_clip, self.cos_logit_clip)
        return logits

    def _select_cos_ids(self, cos_logits, deterministic=False):
        if deterministic:
            return torch.topk(cos_logits, k=self.cos_top_k, dim=-1).indices

        probs = torch.softmax(cos_logits, dim=-1)
        flat_probs = probs.reshape(-1, self.sub_agents)
        return torch.multinomial(
            flat_probs,
            num_samples=self.cos_top_k,
            replacement=False,
        ).view(cos_logits.shape[0], self.sub_agents, self.cos_top_k)

    def _cos_log_prob_for_ids(self, cos_logits, cos_ids):
        masked_logits = cos_logits
        step_log_probs = []
        for step in range(cos_ids.shape[-1]):
            dist = Categorical(logits=masked_logits.reshape(-1, self.sub_agents))
            step_ids = cos_ids[..., step].reshape(-1)
            step_log_probs.append(dist.log_prob(step_ids).view(cos_logits.shape[0], self.sub_agents))
            masked_logits = masked_logits.scatter(-1, cos_ids[..., step:step + 1], -1e9)
        return torch.stack(step_log_probs, dim=-1).sum(dim=-1)

    def _cos_team_context(self, cos_repr, cos_ids):
        batch_size, _, context_dim = cos_repr.shape
        source = cos_repr.unsqueeze(1).expand(-1, self.sub_agents, -1, -1)
        gather_index = cos_ids.unsqueeze(-1).expand(-1, -1, -1, context_dim)
        return torch.gather(source, dim=2, index=gather_index).mean(dim=2)

    def _meta_with_cos(self, state_tensor, base_meta, cos_ids=None, deterministic_cos=False):
        if not self.cos_enabled:
            return base_meta, None, None, None, None

        cos_repr = self._cos_representations(state_tensor, base_meta)
        cos_logits = self._cos_logits(cos_repr)
        if cos_ids is None:
            cos_ids = self._select_cos_ids(cos_logits, deterministic=deterministic_cos)
        cos_log_prob = self._cos_log_prob_for_ids(cos_logits, cos_ids)
        cos_dist = Categorical(logits=cos_logits.reshape(-1, self.sub_agents))
        cos_entropy = cos_dist.entropy().view(state_tensor.shape[0], self.sub_agents).mean()
        cos_probs = torch.softmax(cos_logits, dim=-1)

        team_context = self._cos_team_context(cos_repr, cos_ids)
        if self.cos_fusion_mode == 'add':
            meta = base_meta + self.cos_team_projector(team_context)
        else:
            meta = self.cos_team_projector(torch.cat([base_meta, team_context], dim=-1))
        return meta, cos_ids, cos_log_prob, cos_entropy, cos_probs

    def _actor_forward(self, state_tensor, theta):
        params = {}
        for name, shape, start, end in self.actor_layout:
            params[name] = theta[..., start:end].view(*theta.shape[:-1], *shape)

        x = torch.einsum('bni,bnoi->bno', state_tensor, params['fc1.weight']) + params['fc1.bias']
        x = self._activate(x)
        x = torch.einsum('bni,bnoi->bno', x, params['fc2.weight']) + params['fc2.bias']
        x = self._activate(x)
        return torch.einsum('bni,bnoi->bno', x, params['fc3.weight']) + params['fc3.bias']

    def _actor_forward_iru_generated(self, state_tensor, theta):
        """Functional counterpart of IRUNetwork.forward for a per-agent generated IRU.

        Every Linear layer (input_embedding, each block's forget/input gate,
        output_head) uses a per-agent weight matrix sliced out of ``theta``,
        exactly like ``_actor_forward`` does for the plain MLP actor. The
        LayerNorm submodules are *not* in ``theta`` -- they are real, shared
        ``nn.Module`` instances on ``self.base_actor`` (see
        ``_build_linear_layout_from_module``), so they are called directly.
        """
        params = {}
        for name, shape, start, end in self.actor_layout:
            params[name] = theta[..., start:end].view(*theta.shape[:-1], *shape)

        def linear(x, prefix):
            weight = params[f'{prefix}.weight']
            bias = params[f'{prefix}.bias']
            return torch.einsum('bni,bnoi->bno', x, weight) + bias

        base = self.base_actor
        context = base.input_norm(linear(state_tensor, 'input_embedding'))

        state_shape = (*context.shape[:-1], base.hidden_dim)
        cell_state = context.new_zeros(state_shape)
        hidden_state = context.new_zeros(state_shape)

        for _ in range(base.thinking_steps):
            residual = hidden_state
            for block_idx, block in enumerate(base.blocks):
                combined = torch.cat([context, hidden_state], dim=-1)
                combined = block.recurrence_norm(combined)
                forget = torch.sigmoid(linear(combined, f'blocks.{block_idx}.forget_gate'))
                candidate = torch.tanh(linear(combined, f'blocks.{block_idx}.input_gate'))
                cell_state = forget * cell_state + (1.0 - forget) * candidate
                hidden_state = torch.tanh(cell_state)
            hidden_state = residual + hidden_state

        return linear(hidden_state, 'output_head')

    def _actor_film_forward(self, policy_state, film_params):
        expected_dim = self.actor_film_param_dim
        if film_params.shape[-1] != expected_dim:
            raise RuntimeError(
                f'FiLM hypernetwork output mismatch: expected={expected_dim}, '
                f'actual={film_params.shape[-1]}'
            )
        if self.actor_arch == 'iru':
            return self.base_actor(
                policy_state,
                film_params=film_params,
                film_scale=self.hyper_film_scale,
            )
        gamma1, beta1, gamma2, beta2 = torch.split(
            film_params,
            [
                self.actor_hidden1,
                self.actor_hidden1,
                self.actor_hidden2,
                self.actor_hidden2,
            ],
            dim=-1,
        )
        hidden = self._activate(self.base_actor.fc1(policy_state))
        hidden = hidden * (1.0 + self.hyper_film_scale * torch.tanh(gamma1))
        hidden = hidden + self.hyper_film_scale * torch.tanh(beta1)
        hidden = self._activate(self.base_actor.fc2(hidden))
        hidden = hidden * (1.0 + self.hyper_film_scale * torch.tanh(gamma2))
        hidden = hidden + self.hyper_film_scale * torch.tanh(beta2)
        return self.base_actor.fc3(hidden)

    def _film_diagnostics(self, film_params):
        with torch.no_grad():
            if self.actor_arch == 'iru':
                gamma_context, beta_context, gamma_hidden, beta_hidden = torch.split(
                    film_params,
                    self.iru_actor_hidden_dim,
                    dim=-1,
                )
                gamma = torch.cat([gamma_context, gamma_hidden], dim=-1)
                beta = torch.cat([beta_context, beta_hidden], dim=-1)
            else:
                gamma1, beta1, gamma2, beta2 = torch.split(
                    film_params,
                    [
                        self.actor_hidden1,
                        self.actor_hidden1,
                        self.actor_hidden2,
                        self.actor_hidden2,
                    ],
                    dim=-1,
                )
                gamma = torch.cat([gamma1, gamma2], dim=-1)
                beta = torch.cat([beta1, beta2], dim=-1)
            return {
                'hyper_adapter_is_film': 1.0,
                'hyper_actor_is_iru': float(self.actor_arch == 'iru'),
                'film_scale': float(self.hyper_film_scale),
                'film_gamma_abs_mean': float(gamma.abs().mean().detach().cpu().item()),
                'film_beta_abs_mean': float(beta.abs().mean().detach().cpu().item()),
                'film_param_dim': float(self.actor_film_param_dim),
            }

    def _generated_value_forward(self, value_input, theta):
        x = value_input
        layer_count = len(self.value_layout)
        for layer_idx, (_, _, weight_start, bias_start, end) in enumerate(self.value_layout):
            out_dim, in_dim = self.value_layout[layer_idx][1]
            weight = theta[..., weight_start:bias_start].view(*theta.shape[:-1], out_dim, in_dim)
            bias = theta[..., bias_start:end].view(*theta.shape[:-1], out_dim)
            x = torch.einsum('bni,bnoi->bno', x, weight) + bias
            if layer_idx < len(self.value_layout) - 1:
                x = self._activate(x)
        return x.squeeze(-1)

    def _value_forward_from_meta(self, value_input, meta, return_residual_diagnostics=False):
        chunk_size = self.value_chunk_size
        if chunk_size <= 0 or meta.shape[1] <= chunk_size:
            if self.hyper_critic_adapter_mode == 'film':
                film_params = self.value_hypernet(meta)
                values = self._value_film_forward(value_input, film_params)
                if return_residual_diagnostics:
                    return values, self._value_film_diagnostics(film_params)
                return values
            if self._use_head_residual():
                generated_head = self.value_hypernet(meta)
                values, scaled_head_delta = self._value_head_residual_forward(
                    value_input,
                    generated_head,
                )
                if return_residual_diagnostics:
                    return values, self._compact_head_diagnostics(
                        'value',
                        self.base_value,
                        scaled_head_delta,
                    )
                return values
            value_theta, value_base, value_delta = self._value_theta_from_meta(meta)
            values = self._generated_value_forward(value_input, value_theta)
            if return_residual_diagnostics:
                return values, self._theta_diagnostics(
                    'value',
                    value_base,
                    value_delta,
                    value_theta,
                )
            return values

        values = []
        diagnostics = []
        for start in range(0, meta.shape[1], chunk_size):
            end = min(start + chunk_size, meta.shape[1])
            if self.hyper_critic_adapter_mode == 'film':
                film_params = self.value_hypernet(meta[:, start:end])
                values.append(
                    self._value_film_forward(
                        value_input[:, start:end],
                        film_params,
                    )
                )
                if return_residual_diagnostics:
                    diagnostics.append(self._value_film_diagnostics(film_params))
                continue
            if self._use_head_residual():
                generated_head = self.value_hypernet(meta[:, start:end])
                chunk_values, scaled_head_delta = self._value_head_residual_forward(
                    value_input[:, start:end],
                    generated_head,
                )
                values.append(chunk_values)
                if return_residual_diagnostics:
                    diagnostics.append(
                        self._compact_head_diagnostics(
                            'value',
                            self.base_value,
                            scaled_head_delta,
                        )
                    )
                continue
            value_theta, value_base, value_delta = self._value_theta_from_meta(meta[:, start:end])
            values.append(
                self._generated_value_forward(
                    value_input[:, start:end],
                    value_theta,
                )
            )
            if return_residual_diagnostics:
                chunk_diagnostics = self._theta_diagnostics(
                    'value',
                    value_base,
                    value_delta,
                    value_theta,
                )
                if chunk_diagnostics:
                    diagnostics.append(chunk_diagnostics)
        values = torch.cat(values, dim=1)
        if return_residual_diagnostics:
            return values, self._mean_diagnostics(diagnostics)
        return values

    def _value_input(self, state_tensor):
        if not self.centralized_critic:
            return state_tensor
        if self.centralized_critic_mode == 'concat':
            global_state = state_tensor.reshape(state_tensor.shape[0], -1)
            return global_state.unsqueeze(1).expand(-1, self.sub_agents, -1)

        if self.centralized_critic_mode == 'graph':
            raise RuntimeError('graph centralized critic bypasses the flat _value_input path')

        global_mean = state_tensor.mean(dim=1, keepdim=True)
        global_std = state_tensor.std(dim=1, unbiased=False, keepdim=True)
        global_max = state_tensor.max(dim=1, keepdim=True).values
        global_min = state_tensor.min(dim=1, keepdim=True).values
        global_context = torch.cat([global_mean, global_std, global_max, global_min], dim=-1)
        global_context = global_context.expand(-1, self.sub_agents, -1)
        return torch.cat([state_tensor, global_context], dim=-1)

    def _policy_value(
        self,
        state_tensor,
        cos_ids=None,
        deterministic_cos=False,
        return_cos=False,
        return_residual_diagnostics=False,
    ):
        base_meta = self._agent_meta(state_tensor.shape[0])
        meta, selected_cos_ids, cos_log_prob, cos_entropy, cos_probs = self._meta_with_cos(
            state_tensor,
            base_meta,
            cos_ids=cos_ids,
            deterministic_cos=deterministic_cos,
        )
        policy_state = self._encode_policy_state(state_tensor)
        actor_theta = None
        actor_base = None
        actor_delta = None
        actor_head_delta = None
        film_params = None
        if self.hyper_adapter_mode == 'none':
            raw_logits = self.base_actor(policy_state)
        elif self.hyper_adapter_mode == 'film':
            film_params = self.actor_hypernet(meta)
            raw_logits = self._actor_film_forward(policy_state, film_params)
        elif self._use_head_residual():
            generated_head = self.actor_hypernet(meta)
            raw_logits, actor_head_delta = self._actor_head_residual_forward(
                policy_state,
                generated_head,
            )
        else:
            actor_theta, actor_base, actor_delta = self._actor_theta_from_meta(meta)
            if self._iru_generated_actor:
                raw_logits = self._actor_forward_iru_generated(policy_state, actor_theta)
            else:
                raw_logits = self._actor_forward(policy_state, actor_theta)
        logits = raw_logits.masked_fill(~self.action_mask.unsqueeze(0), -1e9)

        collect_adapter_diagnostics = bool(
            return_residual_diagnostics
            and self.hyper_residual_log_diagnostics
            and (
                self.hyper_residual
                or self.hyper_adapter_mode == 'film'
                or self.hyper_critic_adapter_mode == 'film'
            )
        )
        if self.graph_critic_enabled:
            # The critic never consumes sampled CoS collaborator IDs.  This
            # keeps V(s) action-independent while the actor may use CoS meta.
            values = self.graph_critic(
                policy_state,
                self.graph_edge_index,
                edge_weight=self.graph_edge_weight,
                meta=base_meta if self.graph_critic_film else None,
            )
            residual_diagnostics = {}
        else:
            if collect_adapter_diagnostics and (
                self.hyper_residual
                or self.hyper_critic_adapter_mode == 'film'
            ):
                values, residual_diagnostics = self._value_forward_from_meta(
                    self._value_input(policy_state),
                    meta,
                    return_residual_diagnostics=True,
                )
            else:
                values = self._value_forward_from_meta(self._value_input(policy_state), meta)
                residual_diagnostics = {}

        if collect_adapter_diagnostics:
            if self.hyper_adapter_mode == 'film':
                residual_diagnostics.update(self._film_diagnostics(film_params))
            elif self._use_head_residual():
                residual_diagnostics.update(
                    self._compact_head_diagnostics(
                        'actor',
                        self.base_actor,
                        actor_head_delta,
                    )
                )
            else:
                residual_diagnostics.update(
                    self._theta_diagnostics('actor', actor_base, actor_delta, actor_theta)
                )
            residual_diagnostics.update(self._policy_diagnostics(meta, raw_logits, values))
        if return_cos:
            if return_residual_diagnostics:
                return (
                    logits,
                    values,
                    selected_cos_ids,
                    cos_log_prob,
                    cos_entropy,
                    cos_probs,
                    residual_diagnostics,
                )
            return logits, values, selected_cos_ids, cos_log_prob, cos_entropy, cos_probs
        if return_residual_diagnostics:
            return logits, values, residual_diagnostics
        return logits, values

    def _build_state_np(self, obs, phase):
        if self.phase:
            if self.one_hot:
                phase_feat = utils.idx2onehot(phase.astype(np.int64), self.action_space.n).astype(np.float32)
            else:
                phase_feat = phase.astype(np.float32)[:, np.newaxis]
            state = np.concatenate([obs, phase_feat], axis=-1)
        else:
            state = obs
        return state.astype(np.float32)

    def _policy_prob_from_np(self, ob, phase, deterministic_cos=False, collect_cos_diagnostics=False):
        state = self._build_state_np(ob, phase)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            collect_residual = bool(
                collect_cos_diagnostics
                and (
                    self.hyper_residual
                    or self.hyper_adapter_mode == 'film'
                    or self.hyper_critic_adapter_mode == 'film'
                )
                and self.hyper_residual_log_diagnostics
            )
            policy_output = self._policy_value(
                state_t,
                deterministic_cos=deterministic_cos,
                return_cos=True,
                return_residual_diagnostics=collect_residual,
            )
            if collect_residual:
                (
                    logits,
                    values,
                    cos_ids,
                    cos_log_prob,
                    cos_entropy,
                    cos_probs,
                    residual_diagnostics,
                ) = policy_output
                if residual_diagnostics:
                    self._residual_episode_diagnostics.append(residual_diagnostics)
            else:
                logits, values, cos_ids, cos_log_prob, cos_entropy, cos_probs = policy_output
            if collect_cos_diagnostics:
                diagnostics = self._cos_diagnostics(cos_ids, cos_probs, cos_entropy)
                if diagnostics:
                    self._cos_episode_diagnostics.append(diagnostics)
            probs = torch.softmax(logits.squeeze(0), dim=-1)
        if cos_ids is not None:
            cos_ids = cos_ids.squeeze(0).cpu()
        if cos_log_prob is not None:
            cos_log_prob = cos_log_prob.squeeze(0).cpu()
        return probs.cpu(), values.squeeze(0).cpu(), cos_ids, cos_log_prob

    def reset(self):
        self._build_generators()
        self._cached_action_prob = None
        self._cached_value = None
        self._cached_cos_ids = None
        self._cached_cos_log_prob = None
        self._last_cos_diagnostics = {}
        self._cos_episode_diagnostics = []
        self._last_residual_diagnostics = {}
        self._residual_episode_diagnostics = []
        self._last_abs_pressure = None

    def get_ob(self):
        obs = []
        for ob_gen in self.ob_generator:
            feature = np.asarray(ob_gen.generate(), dtype=np.float32) / self.vehicle_max
            if feature.shape[-1] < self.ob_length:
                feature = np.pad(feature, (0, self.ob_length - feature.shape[-1]))
            elif feature.shape[-1] > self.ob_length:
                feature = feature[: self.ob_length]
            obs.append(feature)
        return np.asarray(obs, dtype=np.float32)

    def get_reward(self):
        rewards = self._queue_waiting_reward()
        if self.reward_mode == 'queue':
            return rewards

        current_abs_pressure = self._current_abs_pressure()
        pressure_reward = -current_abs_pressure / self.pressure_norms
        if self.reward_mode == 'pressure_abs':
            rewards = pressure_reward
        else:
            rewards = rewards + self.pressure_balance_coef * pressure_reward

        if self.pressure_release_coef > 0.0:
            if self._last_abs_pressure is None:
                release_reward = np.zeros_like(current_abs_pressure)
            else:
                release_reward = (self._last_abs_pressure - current_abs_pressure) / self.pressure_norms
            rewards = rewards + self.pressure_release_coef * release_reward

        self._last_abs_pressure = current_abs_pressure
        return np.asarray(rewards, dtype=np.float32)

    def _queue_waiting_reward(self):
        rewards = []
        for reward_gen in self.reward_generator:
            reward = np.asarray(reward_gen.generate(), dtype=np.float32)
            rewards.append(float(np.mean(reward)))
        return np.asarray(rewards, dtype=np.float32)

    def _current_abs_pressure(self):
        lane_count = self.world.get_info('lane_count')
        current_abs_pressure = []
        for in_lanes, out_lanes in self.pressure_lanes:
            pressure = 0.0
            for lane_id in in_lanes:
                pressure += self._lane_count_value(lane_count, lane_id)
            for lane_id in out_lanes:
                pressure -= self._lane_count_value(lane_count, lane_id)
            current_abs_pressure.append(abs(pressure))
        return np.asarray(current_abs_pressure, dtype=np.float32)

    def get_phase(self):
        phase = []
        for phase_gen in self.phase_generator:
            cur_phase = np.asarray(phase_gen.generate()).reshape(-1)
            phase.append(int(cur_phase[0]))
        phase = np.asarray(phase, dtype=np.int64)
        return np.minimum(np.maximum(phase, 0), self.phase_lengths - 1)

    def get_queue(self):
        queue = []
        for queue_gen in self.queue_generator:
            queue.append(float(np.sum(queue_gen.generate())))
        return np.asarray(queue, dtype=np.float32)

    def get_delay(self):
        delay = []
        for delay_gen in self.delay_generator:
            delay.append(float(np.mean(delay_gen.generate())))
        return np.asarray(delay, dtype=np.float32)

    def sample(self):
        return np.asarray(
            [np.random.randint(0, max(1, int(self.phase_lengths[idx]))) for idx in range(self.sub_agents)],
            dtype=np.int64,
        )

    def get_action(self, ob, phase, test=False):
        deterministic_cos = bool(test and self.cos_deterministic_eval)
        probs, values, cos_ids, cos_log_prob = self._policy_prob_from_np(
            ob,
            phase,
            deterministic_cos=deterministic_cos,
            collect_cos_diagnostics=True,
        )
        self._cached_action_prob = probs
        self._cached_value = values.numpy()
        self._cached_cos_ids = None if cos_ids is None else cos_ids.numpy()
        self._cached_cos_log_prob = None if cos_log_prob is None else cos_log_prob.numpy()
        probs_np = probs.numpy()

        if test:
            if self.test_action_mode == 'sample':
                return self._sample_actions_from_probs(
                    probs_np,
                    temperature=self.test_temperature,
                )
            return self._greedy_actions_from_probs(probs_np)

        return self._sample_actions_from_probs(probs_np)

    def _greedy_actions_from_probs(self, probs_np):
        actions = []
        for idx in range(self.sub_agents):
            valid_dim = max(1, int(self.phase_lengths[idx]))
            actions.append(int(np.argmax(probs_np[idx, :valid_dim])))
        return np.asarray(actions, dtype=np.int64)

    def _sample_actions_from_probs(self, probs_np, temperature=1.0):
        actions = []
        for idx in range(self.sub_agents):
            valid_dim = max(1, int(self.phase_lengths[idx]))
            prob = probs_np[idx, :valid_dim].astype(np.float64)
            if temperature != 1.0:
                prob = np.power(np.clip(prob, 1e-12, 1.0), 1.0 / temperature)
            prob_sum = prob.sum()
            if prob_sum <= 1e-8 or not np.isfinite(prob_sum):
                actions.append(np.random.randint(0, valid_dim))
            else:
                actions.append(np.random.choice(valid_dim, p=prob / prob_sum))
        return np.asarray(actions, dtype=np.int64)

    def get_action_prob(self, ob, phase):
        if self._cached_action_prob is not None:
            cached = self._cached_action_prob
            self._cached_action_prob = None
            return cached
        probs, values, cos_ids, cos_log_prob = self._policy_prob_from_np(ob, phase)
        self._cached_value = values.numpy()
        self._cached_cos_ids = None if cos_ids is None else cos_ids.numpy()
        self._cached_cos_log_prob = None if cos_log_prob is None else cos_log_prob.numpy()
        return probs

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        state = self._build_state_np(np.asarray(last_obs, dtype=np.float32), np.asarray(last_phase, dtype=np.int64))
        next_state = self._build_state_np(np.asarray(obs, dtype=np.float32), np.asarray(cur_phase, dtype=np.int64))
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)

        if isinstance(actions_prob, torch.Tensor):
            probs = actions_prob.detach().cpu().numpy()
        else:
            probs = np.asarray(actions_prob, dtype=np.float32)
        chosen_prob = probs[np.arange(self.sub_agents), actions]
        old_log_prob = np.log(np.clip(chosen_prob, 1e-8, 1.0)).astype(np.float32)

        if self._cached_value is None:
            _, values, cos_ids, cos_log_prob = self._policy_prob_from_np(last_obs, last_phase)
            old_value = values.numpy().astype(np.float32)
            self._cached_cos_ids = None if cos_ids is None else cos_ids.numpy()
            self._cached_cos_log_prob = None if cos_log_prob is None else cos_log_prob.numpy()
        else:
            old_value = np.asarray(self._cached_value, dtype=np.float32)
        self._cached_value = None

        if self.cos_enabled:
            if self._cached_cos_ids is None or self._cached_cos_log_prob is None:
                _, _, cos_ids, cos_log_prob = self._policy_prob_from_np(last_obs, last_phase)
                self._cached_cos_ids = cos_ids.numpy()
                self._cached_cos_log_prob = cos_log_prob.numpy()
            old_cos_ids = np.asarray(self._cached_cos_ids, dtype=np.int64)
            old_cos_log_prob = np.asarray(self._cached_cos_log_prob, dtype=np.float32)
        else:
            old_cos_ids = np.zeros((self.sub_agents, 1), dtype=np.int64)
            old_cos_log_prob = np.zeros((self.sub_agents,), dtype=np.float32)
        self._cached_cos_ids = None
        self._cached_cos_log_prob = None

        if np.isscalar(done):
            done_arr = np.full((self.sub_agents,), float(done), dtype=np.float32)
        else:
            done_arr = np.asarray(done, dtype=np.float32).reshape(-1)
            if done_arr.shape[0] != self.sub_agents:
                done_arr = np.full((self.sub_agents,), float(done_arr[0]), dtype=np.float32)

        self.rollout_buffer.append(
            (
                state,
                next_state,
                actions,
                rewards,
                done_arr,
                old_log_prob,
                old_value,
                old_cos_ids,
                old_cos_log_prob,
            )
        )
        self._transitions_since_update += 1

    def _rollout_tensors(self, rollout):
        (
            states,
            next_states,
            actions,
            rewards,
            dones,
            old_log_probs,
            old_values,
            old_cos_ids,
            old_cos_log_probs,
        ) = zip(*rollout)
        return (
            torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(next_states), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(actions), dtype=torch.long, device=self.device),
            torch.tensor(np.asarray(rewards), dtype=torch.float32, device=self.device) * self.reward_scale,
            torch.tensor(np.asarray(dones), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(old_log_probs), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(old_values), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(old_cos_ids), dtype=torch.long, device=self.device),
            torch.tensor(np.asarray(old_cos_log_probs), dtype=torch.float32, device=self.device),
        )

    def _compute_gae(self, rewards, dones, old_values, next_states):
        with torch.no_grad():
            _, last_value = self._policy_value(next_states[-1:].detach(), deterministic_cos=True)
            last_value = last_value.squeeze(0)

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros((self.sub_agents,), dtype=torch.float32, device=self.device)
        for step in reversed(range(rewards.shape[0])):
            next_nonterminal = 1.0 - dones[step]
            next_value = last_value if step == rewards.shape[0] - 1 else old_values[step + 1]
            delta = rewards[step] + self.gamma * next_value * next_nonterminal - old_values[step]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            advantages[step] = last_gae
        returns = advantages + old_values
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return advantages.detach(), returns.detach()

    def _policy_surrogate_loss(self, new_log_prob, old_log_prob, advantage):
        ratio = torch.exp(new_log_prob - old_log_prob)
        if self.policy_objective == 'spo':
            policy_objective = (
                ratio * advantage
                - advantage.abs() * (ratio - 1.0).pow(2) / (2.0 * self.spo_eps)
            )
            return -policy_objective.mean()

        policy_loss_1 = ratio * advantage
        policy_loss_2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
        return -torch.min(policy_loss_1, policy_loss_2).mean()

    def _cos_regularization_loss(self, cos_probs):
        if cos_probs is None:
            return 0.0

        reg_loss = cos_probs.new_tensor(0.0)
        if self.cos_diag_coef != 0.0:
            diag_prob = torch.diagonal(cos_probs, dim1=-2, dim2=-1).mean()
            reg_loss = reg_loss - self.cos_diag_coef * diag_prob
        if self.cos_symmetry_coef != 0.0:
            sym_loss = (cos_probs - cos_probs.transpose(-1, -2)).pow(2).mean()
            reg_loss = reg_loss + self.cos_symmetry_coef * sym_loss
        return reg_loss

    def _cos_diagnostics(self, cos_ids, cos_probs, cos_entropy):
        if not self.cos_log_diagnostics or cos_ids is None or cos_probs is None:
            return {}

        with torch.no_grad():
            agent_ids = torch.arange(self.sub_agents, device=cos_ids.device).view(1, self.sub_agents, 1)
            selected_self = cos_ids == agent_ids
            diagnostics = {
                'cos_entropy': float(cos_entropy.detach().cpu().item()) if cos_entropy is not None else np.nan,
                'cos_self_selection_rate': float(selected_self.any(dim=-1).float().mean().detach().cpu().item()),
                'cos_diag_mass': float(
                    torch.diagonal(cos_probs, dim1=-2, dim2=-1).mean().detach().cpu().item()
                ),
                'cos_symmetry_loss': float(
                    (cos_probs - cos_probs.transpose(-1, -2)).pow(2).mean().detach().cpu().item()
                ),
            }

            row_index = agent_ids.expand(cos_ids.shape[0], self.sub_agents, cos_ids.shape[-1])
            if self.cos_pairwise_hops is not None:
                selected_hops = self.cos_pairwise_hops.to(cos_ids.device)[row_index, cos_ids]
                diagnostics['cos_avg_selected_hop'] = float(selected_hops.mean().detach().cpu().item())
            if self.cos_pairwise_distances is not None:
                selected_distances = self.cos_pairwise_distances.to(cos_ids.device)[row_index, cos_ids]
                diagnostics['cos_avg_selected_distance'] = float(
                    selected_distances.mean().detach().cpu().item()
                )

        return diagnostics

    @staticmethod
    def _mean_cos_diagnostics(diagnostics):
        if not diagnostics:
            return {}
        keys = sorted({key for item in diagnostics for key in item})
        averaged = {}
        for key in keys:
            values = [
                float(item[key])
                for item in diagnostics
                if key in item and np.isfinite(float(item[key]))
            ]
            if values:
                averaged[key] = float(np.mean(values))
        return averaged

    def get_cos_diagnostics(self):
        if not self.cos_log_diagnostics:
            return {}
        return dict(self._last_cos_diagnostics)

    def get_cos_episode_diagnostics(self):
        if not self.cos_log_diagnostics:
            return {}
        return self._mean_cos_diagnostics(self._cos_episode_diagnostics)

    def get_residual_diagnostics(self):
        if (
            not (
                self.hyper_residual
                or self.hyper_adapter_mode == 'film'
                or self.hyper_critic_adapter_mode == 'film'
            )
            or not self.hyper_residual_log_diagnostics
        ):
            return {}
        return dict(self._last_residual_diagnostics)

    def get_residual_episode_diagnostics(self):
        if (
            not (
                self.hyper_residual
                or self.hyper_adapter_mode == 'film'
                or self.hyper_critic_adapter_mode == 'film'
            )
            or not self.hyper_residual_log_diagnostics
        ):
            return {}
        return self._mean_diagnostics(self._residual_episode_diagnostics)

    def train(self):
        if self._transitions_since_update < self.ppo_rollout_steps:
            self._last_cos_diagnostics = {}
            self._last_residual_diagnostics = {}
            return 0.0

        rollout = list(self.rollout_buffer)
        self.rollout_buffer.clear()
        self._transitions_since_update = 0

        (
            state_t,
            next_state_t,
            action_t,
            reward_t,
            done_t,
            old_log_prob_t,
            old_value_t,
            old_cos_ids_t,
            old_cos_log_prob_t,
        ) = self._rollout_tensors(rollout)
        advantages_t, returns_t = self._compute_gae(reward_t, done_t, old_value_t, next_state_t)

        num_steps = state_t.shape[0]
        step_batch_size = max(1, min(num_steps, self.ppo_minibatch_size // max(1, self.sub_agents)))
        losses = []
        cos_diagnostics = []
        residual_diagnostics = []
        collect_residual = bool(
            (
                self.hyper_residual
                or self.hyper_adapter_mode == 'film'
                or self.hyper_critic_adapter_mode == 'film'
            )
            and self.hyper_residual_log_diagnostics
        )

        for _ in range(self.ppo_epochs):
            order = np.random.permutation(num_steps)
            for start in range(0, num_steps, step_batch_size):
                batch_idx = torch.tensor(order[start:start + step_batch_size], dtype=torch.long, device=self.device)
                b_state = state_t.index_select(0, batch_idx)
                b_action = action_t.index_select(0, batch_idx)
                b_old_log_prob = old_log_prob_t.index_select(0, batch_idx)
                b_old_value = old_value_t.index_select(0, batch_idx)
                b_advantage = advantages_t.index_select(0, batch_idx)
                b_return = returns_t.index_select(0, batch_idx)
                b_old_cos_ids = old_cos_ids_t.index_select(0, batch_idx)
                b_old_cos_log_prob = old_cos_log_prob_t.index_select(0, batch_idx)

                if self.cos_enabled:
                    policy_output = self._policy_value(
                        b_state,
                        cos_ids=b_old_cos_ids,
                        return_cos=True,
                        return_residual_diagnostics=collect_residual,
                    )
                    if collect_residual:
                        (
                            logits,
                            values,
                            _,
                            new_cos_log_prob,
                            cos_entropy,
                            cos_probs,
                            batch_residual_diagnostics,
                        ) = policy_output
                    else:
                        logits, values, _, new_cos_log_prob, cos_entropy, cos_probs = policy_output
                        batch_residual_diagnostics = {}
                else:
                    if collect_residual:
                        logits, values, batch_residual_diagnostics = self._policy_value(
                            b_state,
                            return_residual_diagnostics=True,
                        )
                    else:
                        logits, values = self._policy_value(b_state)
                        batch_residual_diagnostics = {}
                    new_cos_log_prob = None
                    cos_entropy = None
                    cos_probs = None
                dist = Categorical(logits=logits.reshape(-1, self.action_space.n))
                new_log_prob = dist.log_prob(b_action.reshape(-1)).view_as(b_action)
                entropy = dist.entropy().mean()

                policy_loss = self._policy_surrogate_loss(new_log_prob, b_old_log_prob, b_advantage)
                cos_reg_loss = 0.0
                if self.cos_enabled:
                    cos_policy_loss = self._policy_surrogate_loss(
                        new_cos_log_prob,
                        b_old_cos_log_prob,
                        b_advantage,
                    )
                    policy_loss = policy_loss + self.cos_policy_coef * cos_policy_loss
                    cos_reg_loss = self._cos_regularization_loss(cos_probs)

                if self.clip_vf is not None and self.clip_vf > 0.0:
                    value_clipped = b_old_value + (values - b_old_value).clamp(-self.clip_vf, self.clip_vf)
                    value_loss = torch.max(
                        (values - b_return).pow(2),
                        (value_clipped - b_return).pow(2),
                    ).mean()
                else:
                    value_loss = (values - b_return).pow(2).mean()
                value_loss = 0.5 * value_loss

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                if self.cos_enabled:
                    loss = loss + cos_reg_loss - self.cos_entropy_coef * cos_entropy
                if not torch.isfinite(loss):
                    continue

                if self.cos_enabled:
                    diagnostics = self._cos_diagnostics(b_old_cos_ids, cos_probs, cos_entropy)
                    if diagnostics:
                        cos_diagnostics.append(diagnostics)
                if collect_residual and batch_residual_diagnostics:
                    residual_diagnostics.append(batch_residual_diagnostics)

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self._optimizer_parameters(), self.grad_clip)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

        if self.cos_enabled:
            self._last_cos_diagnostics = self._mean_cos_diagnostics(cos_diagnostics)
        if collect_residual:
            self._last_residual_diagnostics = self._mean_diagnostics(residual_diagnostics)

        return float(np.mean(losses)) if losses else 0.0

    def _optimizer_parameters(self):
        params = []
        if self.actor_hypernet is not None:
            params.extend(self.actor_hypernet.parameters())
        if self.value_hypernet is not None:
            params.extend(self.value_hypernet.parameters())
        if self.graph_critic is not None:
            params.extend(self.graph_critic.parameters())
        if self.movement_encoder is not None:
            params.extend(self.movement_encoder.parameters())
        if self.base_actor_trainable:
            params.extend(self.base_actor.parameters())
        if self.base_value_trainable and self.base_value is not None:
            params.extend(self.base_value.parameters())
        if isinstance(self.agent_embeddings, nn.Parameter):
            params.append(self.agent_embeddings)
        if self.topology_encoder is not None:
            params.extend(self.topology_encoder.parameters())
        if self.cos_enabled:
            params.extend(self._cos_parameters())
        unique_params = []
        seen = set()
        for param in params:
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            unique_params.append(param)
        return unique_params

    def update_target_network(self):
        pass

    @staticmethod
    def _tensor_fingerprint(tensor):
        if tensor is None:
            return None
        array = tensor.detach().cpu().contiguous().numpy()
        return hashlib.sha256(array.tobytes()).hexdigest()

    def _architecture_signature(self):
        return {
            'version': 4,
            'node_count': int(self.sub_agents),
            'action_dim': int(self.action_space.n),
            'phase_lengths': self.phase_lengths.tolist(),
            'raw_state_dim': int(self.raw_state_dim),
            'policy_input_dim': int(self.policy_input_dim),
            'phase': self.phase,
            'one_hot': self.one_hot,
            'vehicle_max': float(self.vehicle_max),
            'activation': self.activation,
            'embedding_mode': self.embedding_mode,
            'meta_dim': int(self.meta_dim),
            'topology_aware_embedding': self.topology_aware_embedding,
            'topology_fingerprint': self._tensor_fingerprint(
                getattr(self, 'registered_topology_features', None)
            ),
            'actor_hypernet_type': str(self.hypernet_type),
            'hyper_actor_arch': self.actor_arch,
            'iru_actor_hidden_dim': (
                int(self.iru_actor_hidden_dim) if self.actor_arch == 'iru' else None
            ),
            'iru_actor_steps': (
                int(self.iru_actor_steps) if self.actor_arch == 'iru' else None
            ),
            'iru_num_blocks': (
                int(self.iru_num_blocks) if self.actor_arch == 'iru' else None
            ),
            'iru_layer_norm': (
                self.iru_layer_norm if self.actor_arch == 'iru' else None
            ),
            'value_hypernet_type': str(self.value_hypernet_type),
            'hyper_head_mode': self.hyper_head_mode,
            'actor_hidden': [int(self.actor_hidden1), int(self.actor_hidden2)],
            'value_hidden': list(self.value_hidden),
            'hyper_adapter_mode': self.hyper_adapter_mode,
            'hyper_critic_adapter_mode': self.hyper_critic_adapter_mode,
            'hyper_film_scale': float(self.hyper_film_scale),
            'hyper_residual': self.hyper_residual,
            'hyper_residual_mode': self.hyper_residual_mode,
            'hyper_residual_actor_scale': float(self.hyper_residual_actor_scale),
            'hyper_residual_value_scale': float(self.hyper_residual_value_scale),
            'hyper_lora_actor_rank': int(self.hyper_lora_actor_rank),
            'hyper_lora_value_rank': int(self.hyper_lora_value_rank),
            'hyper_lora_bias': self.hyper_lora_bias,
            'movement_encoder_enabled': self.movement_encoder_enabled,
            'movement_state_features': list(self.state_features),
            'movement_encoder_dim': int(self.policy_input_dim),
            'movement_token_dim': int(self.movement_token_dim),
            'movement_encoder_heads': int(self.movement_encoder_heads),
            'movement_encoder_layers': int(self.movement_encoder_layers),
            'movement_encoder_ff_dim': int(self.movement_encoder_ff_dim),
            'movement_encoder_dropout': float(self.movement_encoder_dropout),
            'movement_token_count': (
                None
                if self.movement_token_mask is None
                else int(self.movement_token_mask.shape[-1])
            ),
            'movement_index_fingerprint': self._tensor_fingerprint(
                self.movement_feature_indices
            ),
            'movement_mask_fingerprint': self._tensor_fingerprint(
                self.movement_token_mask
            ),
            'movement_phase_fingerprint': self._tensor_fingerprint(
                self.movement_phase_availability
            ),
            'movement_turn_fingerprint': self._tensor_fingerprint(
                self.movement_turn_features
            ),
            'movement_source_fingerprint': self._tensor_fingerprint(
                self.movement_source_position
            ),
            'movement_destination_fingerprint': self._tensor_fingerprint(
                self.movement_position
            ),
            'graph_critic_enabled': self.graph_critic_enabled,
            'graph_critic_hidden_dim': int(self.graph_critic_hidden_dim),
            'graph_critic_layers': int(self.graph_critic_layers),
            'graph_critic_heads': int(self.graph_critic_heads),
            'graph_critic_dropout': float(self.graph_critic_dropout),
            'graph_critic_use_edge_weight': self.graph_critic_use_edge_weight,
            'graph_critic_edge_weight_scale': float(self.graph_critic_edge_weight_scale),
            'graph_critic_global_pool': self.graph_critic_global_pool,
            'graph_critic_film': self.graph_critic_film,
            'graph_critic_film_hidden': list(self.graph_critic_film_hidden),
            'graph_critic_film_scale': float(self.graph_critic_film_scale),
            'graph_edge_count': (
                None if self.graph_edge_index is None else int(self.graph_edge_index.shape[1])
            ),
            'graph_edge_fingerprint': self._tensor_fingerprint(self.graph_edge_index),
            'graph_weight_fingerprint': (
                self._tensor_fingerprint(self.graph_edge_weight)
                if self.graph_critic_use_edge_weight
                else None
            ),
            'graph_message_direction': self.graph_message_direction,
            'centralized_critic_mode': self.centralized_critic_mode,
        }

    def _validate_checkpoint_architecture(self, checkpoint):
        expected = self._architecture_signature()
        actual = checkpoint.get('architecture')
        requires_versioned_architecture = bool(
            self.actor_arch == 'iru'
            or self.hyper_adapter_mode == 'none'
            or self.hyper_adapter_mode == 'film'
            or self.hyper_critic_adapter_mode == 'film'
            or self.movement_encoder_enabled
            or self.graph_critic_enabled
        )
        if actual is None:
            if requires_versioned_architecture:
                raise RuntimeError(
                    'checkpoint predates the IRU/movement/graph/FiLM architecture and is '
                    'not compatible with the selected model config'
                )
            return
        actual = dict(actual)
        if int(actual.get('version', 2)) < 3:
            # Version 2 checkpoints always used the MLP actor.  Supplying the
            # new defaults keeps those checkpoints loadable without weakening
            # validation for IRU checkpoints.
            actual['version'] = 3
            actual.setdefault('hyper_actor_arch', 'mlp')
            actual.setdefault('iru_actor_hidden_dim', None)
            actual.setdefault('iru_actor_steps', None)
            actual.setdefault('iru_num_blocks', None)
            actual.setdefault('iru_layer_norm', None)
        if int(actual.get('version', 3)) < 4:
            # Version 3 always used the original generated-weight critic.
            actual['version'] = 4
            actual.setdefault('hyper_critic_adapter_mode', 'generated')
        for key in expected:
            if actual.get(key) != expected[key]:
                raise RuntimeError(
                    f'checkpoint architecture mismatch for {key}: '
                    f'checkpoint={actual.get(key)!r}, config={expected[key]!r}'
                )

    def save_model(self, e=0):
        model_dir = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        payload = {
            'actor_hypernet': (
                None if self.actor_hypernet is None else self.actor_hypernet.state_dict()
            ),
            'value_hypernet': (
                None if self.value_hypernet is None else self.value_hypernet.state_dict()
            ),
            'optimizer': self.optimizer.state_dict(),
            'architecture_version': 4,
            'architecture': self._architecture_signature(),
            'embedding_mode': self.embedding_mode,
            'agent_embeddings': self.agent_embeddings.detach().cpu(),
            'hyper_residual': self.hyper_residual,
            'hyper_residual_mode': self.hyper_residual_mode,
            'hyper_lora_actor_rank': self.hyper_lora_actor_rank,
            'hyper_lora_value_rank': self.hyper_lora_value_rank,
            'hyper_lora_bias': self.hyper_lora_bias,
        }
        if self.base_actor_trainable:
            payload['base_actor'] = self.base_actor.state_dict()
        if self.base_value_trainable and self.base_value is not None:
            payload['base_value'] = self.base_value.state_dict()
        if self.movement_encoder is not None:
            payload['movement_encoder'] = self.movement_encoder.state_dict()
            payload['movement_token_count'] = int(self.movement_token_mask.shape[-1])
        if self.graph_critic is not None:
            payload['graph_critic'] = self.graph_critic.state_dict()
        if self.topology_encoder is not None:
            payload['topology_encoder'] = self.topology_encoder.state_dict()
            payload['topology_feature_names'] = self.topology_feature_names
        if self.cos_enabled:
            payload['cos_state_encoder'] = (
                None if self.cos_state_encoder is None else self.cos_state_encoder.state_dict()
            )
            payload['cos_meta_encoder'] = (
                None if self.cos_meta_encoder is None else self.cos_meta_encoder.state_dict()
            )
            payload['cos_selector'] = self.cos_selector.state_dict()
            payload['cos_team_projector'] = self.cos_team_projector.state_dict()
        torch.save(payload, os.path.join(model_dir, f'{e}_{self.rank}.pt'))

    def load_model(self, e=0):
        model_dir = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        checkpoint = torch.load(os.path.join(model_dir, f'{e}_{self.rank}.pt'), map_location=self.device)
        self._validate_checkpoint_architecture(checkpoint)
        if self.actor_hypernet is not None:
            actor_hypernet_state = checkpoint.get('actor_hypernet')
            if actor_hypernet_state is None:
                raise RuntimeError('checkpoint does not contain the configured actor hypernetwork')
            self.actor_hypernet.load_state_dict(actor_hypernet_state)
        if self.value_hypernet is not None:
            value_state = checkpoint.get('value_hypernet')
            if value_state is None:
                raise RuntimeError('checkpoint does not contain the configured value hypernetwork')
            self.value_hypernet.load_state_dict(value_state)
        if self.movement_encoder is not None:
            movement_state = checkpoint.get('movement_encoder')
            if movement_state is None:
                raise RuntimeError('checkpoint does not contain the configured movement encoder')
            self.movement_encoder.load_state_dict(movement_state)
        if self.graph_critic is not None:
            graph_state = checkpoint.get('graph_critic')
            if graph_state is None:
                raise RuntimeError('checkpoint does not contain the configured graph critic')
            self.graph_critic.load_state_dict(graph_state)
        if 'base_actor' in checkpoint:
            self.base_actor.load_state_dict(checkpoint['base_actor'])
        elif self.base_actor_trainable:
            raise RuntimeError(
                f'{self.hyper_adapter_mode} checkpoint does not contain its shared base actor'
            )
        if self.base_value is not None and 'base_value' in checkpoint:
            self.base_value.load_state_dict(checkpoint['base_value'])
        if isinstance(self.agent_embeddings, nn.Parameter) and 'agent_embeddings' in checkpoint:
            self.agent_embeddings.data.copy_(checkpoint['agent_embeddings'].to(self.device))
        if self.topology_encoder is not None and 'topology_encoder' in checkpoint:
            self.topology_encoder.load_state_dict(checkpoint['topology_encoder'])
        if self.cos_enabled:
            if self.cos_state_encoder is not None and checkpoint.get('cos_state_encoder') is not None:
                self.cos_state_encoder.load_state_dict(checkpoint['cos_state_encoder'])
            if self.cos_meta_encoder is not None and checkpoint.get('cos_meta_encoder') is not None:
                self.cos_meta_encoder.load_state_dict(checkpoint['cos_meta_encoder'])
            if 'cos_selector' in checkpoint:
                self.cos_selector.load_state_dict(checkpoint['cos_selector'])
            if 'cos_team_projector' in checkpoint:
                self.cos_team_projector.load_state_dict(checkpoint['cos_team_projector'])
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError:
                if checkpoint.get('architecture_version', 1) >= 2:
                    raise


@Registry.register_model('hyperlight_mappo')
class HyperLightMAPPOAgent(HyperLightPPOAgent):
    """
    MAPPO-style registration. The behavior is controlled by config, especially
    centralized_critic=True in configs/tsc/hyperlight_mappo.yml.
    """

    pass


@Registry.register_model('hyperlight_graph_mappo')
class HyperLightGraphMAPPOAgent(HyperLightMAPPOAgent):
    """Movement-token actor with a directed graph centralized critic."""

    pass


@Registry.register_model('hyperlight_mappo_cos')
class HyperLightMAPPOCoSAgent(HyperLightMAPPOAgent):
    """
    MAPPO-style HyperLight with CoS enabled by config.
    """

    pass


@Registry.register_model('hyperlight_maspo')
class HyperLightMASPOAgent(HyperLightPPOAgent):
    """
    Multi-Agent SPO (Soft Policy Optimization) registration. The behavior is
    controlled by config, with policy_objective=spo in configs/tsc/hyperlight_maspo.yml.
    """

    pass
