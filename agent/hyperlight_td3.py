from collections import deque
import os
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from .actor import BaseActor
from .hypernetwork import (
    build_generated_param_init_config,
    build_hypernetwork,
)
from .rl_agent import RLAgent
from . import utils
from common.registry import Registry
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator


class HyperGeneratedQNetwork(nn.Module):
    """
    Agent-conditioned Q network.

    The target Q MLP receives TSC state/action features, while its parameters
    are generated from the per-agent embedding by a HyperMARL-style hypernet.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        meta_dim,
        hidden_dims,
        hyper_hidden,
        hypernet_type='mlp',
        head_mode='flat',
        use_bias=True,
        head_init_gain=1.0,
        dropout=0.0,
        chunk_size=0,
        rf_init_config=None,
        activation='relu',
    ):
        super().__init__()
        self.dims = [int(state_dim) + int(action_dim)] + [int(v) for v in hidden_dims] + [1]
        self.layout = self._build_layout_from_dims(self.dims)
        self.param_dim = self.layout[-1][-1]
        self.activation = str(activation or 'relu').lower()
        self.chunk_size = int(chunk_size or 0)

        rf_init_config = rf_init_config or {}
        self.hypernet = build_hypernetwork(
            hypernet_type,
            int(meta_dim),
            [int(v) for v in hyper_hidden],
            self.param_dim,
            dropout=float(dropout),
            target_layout=self.layout,
            head_mode=head_mode,
            use_bias=bool(use_bias),
            head_init_gain=float(head_init_gain),
            **rf_init_config,
        )

    @staticmethod
    def _build_layout_from_dims(dims):
        layout = []
        offset = 0
        for layer_idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            weight_numel = int(out_dim) * int(in_dim)
            bias_numel = int(out_dim)
            layout.append(
                (
                    f'layer{layer_idx}',
                    (int(out_dim), int(in_dim)),
                    offset,
                    offset + weight_numel,
                    offset + weight_numel + bias_numel,
                )
            )
            offset += weight_numel + bias_numel
        return layout

    def _activate(self, x):
        if self.activation == 'tanh':
            return torch.tanh(x)
        return F.relu(x)

    def _forward_generated(self, x, theta):
        for layer_idx, (_, (out_dim, in_dim), weight_start, bias_start, end) in enumerate(self.layout):
            weight = theta[..., weight_start:bias_start].view(*theta.shape[:-1], out_dim, in_dim)
            bias = theta[..., bias_start:end].view(*theta.shape[:-1], out_dim)
            if x.dim() == 3:
                x = torch.einsum('bni,bnoi->bno', x, weight) + bias
            else:
                x = torch.einsum('mi,moi->mo', x, weight) + bias
            if layer_idx < len(self.layout) - 1:
                x = self._activate(x)
        return x

    def forward(self, state, action, meta):
        x = torch.cat([state, action], dim=-1)
        batch_size, n_agents, _ = x.shape
        if self.chunk_size <= 0 or batch_size * n_agents <= self.chunk_size:
            theta = self.hypernet(meta)
            return self._forward_generated(x, theta).squeeze(-1)

        x_flat = x.reshape(batch_size * n_agents, -1)
        meta_flat = meta.reshape(batch_size * n_agents, -1)
        outputs = []
        for start in range(0, x_flat.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, x_flat.shape[0])
            theta = self.hypernet(meta_flat[start:end])
            outputs.append(self._forward_generated(x_flat[start:end], theta))
        return torch.cat(outputs, dim=0).view(batch_size, n_agents)


class HyperTwinTD3Critic(nn.Module):
    """
    TD3 twin critics with HyperMARL-generated local Q functions.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        meta_dim,
        hidden_dims,
        hyper_hidden,
        hypernet_type='mlp',
        head_mode='flat',
        use_bias=True,
        head_init_gain=1.0,
        dropout=0.0,
        chunk_size=0,
        rf_init_config=None,
        activation='relu',
    ):
        super().__init__()
        self.q1_net = HyperGeneratedQNetwork(
            state_dim,
            action_dim,
            meta_dim,
            hidden_dims,
            hyper_hidden,
            hypernet_type=hypernet_type,
            head_mode=head_mode,
            use_bias=use_bias,
            head_init_gain=head_init_gain,
            dropout=dropout,
            chunk_size=chunk_size,
            rf_init_config=rf_init_config,
            activation=activation,
        )
        self.q2_net = HyperGeneratedQNetwork(
            state_dim,
            action_dim,
            meta_dim,
            hidden_dims,
            hyper_hidden,
            hypernet_type=hypernet_type,
            head_mode=head_mode,
            use_bias=use_bias,
            head_init_gain=head_init_gain,
            dropout=dropout,
            chunk_size=chunk_size,
            rf_init_config=rf_init_config,
            activation=activation,
        )

    def forward(self, state, action, meta, reduce=False):
        q1 = self.q1_net(state, action, meta)
        q2 = self.q2_net(state, action, meta)
        if reduce:
            return q1.mean(dim=1), q2.mean(dim=1)
        return q1, q2

    def q1(self, state, action, meta, reduce=False):
        q1 = self.q1_net(state, action, meta)
        return q1.mean(dim=1) if reduce else q1


@Registry.register_model('hyper_td3')
@Registry.register_model('hyperlight_td3')
class HyperLightTD3Agent(RLAgent):
    """
    HyperMARL-style TD3 controller for traffic signal control.

    This branch keeps the paper/repository spirit: agent embeddings condition
    hypernetworks that generate agent-specific actor and critic parameters.
    It deliberately excludes surrogate dynamics and MB-HyperMARL additions.
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
        self.tau = float(cfg.get('tau', 0.005))
        self.batch_size = max(1, int(cfg.get('batch_size', 128)))
        self.reward_scale = float(cfg.get('reward_scale', 1.0))
        self.use_mplight_reward = bool(cfg.get('use_mplight_reward', True))
        self.grad_clip = float(cfg.get('grad_clip', 0.5))
        self.policy_delay = max(1, int(cfg.get('policy_delay', 2)))
        self.actor_warmup_steps = max(0, int(cfg.get('actor_warmup_steps', 0)))
        self.actor_entropy_coef = float(cfg.get('actor_entropy_coef', 0.0))
        self.target_policy_noise = float(cfg.get('target_policy_noise', 0.02))
        self.target_noise_clip = float(cfg.get('target_noise_clip', 0.05))
        self.huber_beta = float(cfg.get('huber_beta', 1.0))
        self.external_target_update = bool(cfg.get('external_target_update', False))
        self.discrete_td3_mode = str(cfg.get('td3_discrete_mode', 'expected_q')).lower()
        if self.discrete_td3_mode not in ('expected_q', 'soft_action'):
            raise ValueError(f"Unknown td3_discrete_mode: {self.discrete_td3_mode}")
        self.epsilon = float(cfg.get('epsilon', 0.5))
        self.epsilon_decay = float(cfg.get('epsilon_decay', 0.9995))
        self.epsilon_min = float(cfg.get('epsilon_min', 0.02))
        self.test_action_mode = str(cfg.get('test_action_mode', 'argmax')).lower()
        if self.test_action_mode == 'stochastic':
            self.test_action_mode = 'sample'
        if self.test_action_mode not in ('argmax', 'sample'):
            raise ValueError(f"Unknown test_action_mode: {self.test_action_mode}")
        self.test_temperature = max(1e-6, float(cfg.get('test_temperature', 1.0)))

        self.activation = str(cfg.get('activation', 'relu')).lower()
        if self.activation not in ('relu', 'tanh'):
            raise ValueError(f"Unknown HyperLight TD3 activation: {self.activation}")
        self.centralized_critic = bool(cfg.get('centralized_critic', False))
        self.centralized_critic_mode = str(cfg.get('centralized_critic_mode', 'pooled')).lower()

        self.actor_hidden1 = int(cfg.get('actor_hidden1', 64))
        self.actor_hidden2 = int(cfg.get('actor_hidden2', 64))
        critic_hidden = cfg.get('critic_hidden', [128, 128])
        if not isinstance(critic_hidden, list):
            critic_hidden = [int(critic_hidden)]
        self.critic_hidden = [int(v) for v in critic_hidden]

        self.hypernet_type = str(cfg.get('hypernet_type', 'mlp')).lower()
        self.actor_hypernet_type = str(cfg.get('actor_hypernet_type', self.hypernet_type)).lower()
        self.critic_hypernet_type = str(cfg.get('critic_hypernet_type', self.hypernet_type)).lower()
        self.hyper_head_mode = str(cfg.get('hyper_head_mode', 'flat')).lower()
        self.hyper_use_bias = bool(cfg.get('hyper_use_bias', True))
        self.hyper_head_init_gain = float(cfg.get('hyper_head_init_gain', 1.0))
        self.actor_rf_init_config = build_generated_param_init_config(
            cfg,
            output_gain_key='hyper_rf_actor_output_gain',
            default_output_gain=0.01,
        )
        self.critic_rf_init_config = build_generated_param_init_config(
            cfg,
            output_gain_key='hyper_rf_critic_output_gain',
            default_output_gain=1.0,
        )
        hyper_hidden = cfg.get('hyper_hidden', [64])
        if not isinstance(hyper_hidden, list):
            hyper_hidden = [int(hyper_hidden)]
        self.hyper_hidden = [int(v) for v in hyper_hidden]
        critic_hyper_hidden = cfg.get('critic_hyper_hidden', self.hyper_hidden)
        if not isinstance(critic_hyper_hidden, list):
            critic_hyper_hidden = [int(critic_hyper_hidden)]
        self.critic_hyper_hidden = [int(v) for v in critic_hyper_hidden]
        self.hyper_dropout = float(cfg.get('hyper_dropout', 0.0))
        self.critic_chunk_size = int(cfg.get('critic_chunk_size', 0))

        self._build_generators()
        self.state_dim = self.ob_length + (self.action_space.n if self.phase and self.one_hot else int(self.phase))
        self.action_dim = self.action_space.n
        self.action_mask = self._build_action_mask().to(self.device)

        if not self.centralized_critic:
            self.critic_state_dim = self.state_dim
            self.critic_action_dim = self.action_dim
        elif self.centralized_critic_mode == 'concat':
            self.critic_state_dim = self.state_dim * self.sub_agents
            self.critic_action_dim = self.action_dim * self.sub_agents
        elif self.centralized_critic_mode == 'pooled':
            self.critic_state_dim = self.state_dim * 5
            self.critic_action_dim = self.action_dim * 5
        else:
            raise ValueError(f"Unknown centralized_critic_mode: {self.centralized_critic_mode}")

        self.embedding_mode = str(cfg.get('agent_embedding_mode', 'learned')).lower()
        if self.embedding_mode == 'learned':
            embedding_dim = int(cfg.get('agent_embedding_dim', min(64, self.sub_agents)))
            self.agent_embeddings = nn.Parameter(torch.empty(self.sub_agents, embedding_dim, device=self.device))
            nn.init.orthogonal_(self.agent_embeddings)
            self.target_agent_embeddings = nn.Parameter(
                torch.empty(self.sub_agents, embedding_dim, device=self.device),
                requires_grad=False,
            )
            self.meta_dim = embedding_dim
        elif self.embedding_mode == 'one_hot':
            self.agent_embeddings = torch.eye(self.sub_agents, dtype=torch.float32, device=self.device)
            self.target_agent_embeddings = self.agent_embeddings
            self.meta_dim = self.sub_agents
        else:
            raise ValueError(f"Unknown agent_embedding_mode: {self.embedding_mode}")

        self.base_actor = BaseActor(
            self.state_dim,
            self.actor_hidden1,
            self.actor_hidden2,
            self.action_dim,
        ).to(self.device)
        for param in self.base_actor.parameters():
            param.requires_grad = False

        self.actor_layout = self._build_layout_from_module(self.base_actor)
        self.actor_param_dim = self.actor_layout[-1][-1]
        self.actor_hypernet = self._build_actor_hypernet(cfg).to(self.device)
        self.target_actor_hypernet = self._build_actor_hypernet(cfg).to(self.device)

        self.critic = self._build_critic(cfg).to(self.device)
        self.target_critic = self._build_critic(cfg).to(self.device)

        actor_params = list(self.actor_hypernet.parameters())
        critic_params = list(self.critic.parameters())
        if isinstance(self.agent_embeddings, nn.Parameter):
            actor_params.append(self.agent_embeddings)
            critic_params.append(self.agent_embeddings)

        self.actor_optimizer = optim.Adam(
            actor_params,
            lr=float(cfg.get('actor_lr', cfg.get('learning_rate', 1e-4))),
            eps=float(cfg.get('adam_eps', 1e-5)),
        )
        self.critic_optimizer = optim.Adam(
            critic_params,
            lr=float(cfg.get('critic_lr', cfg.get('learning_rate', 1e-4))),
            eps=float(cfg.get('adam_eps', 1e-5)),
        )

        buffer_size = int(trainer_cfg.get('buffer_size', cfg.get('buffer_size', 20000)))
        self.replay_buffer = deque(maxlen=max(buffer_size, self.batch_size))
        self.train_step = 0
        self._cached_action_prob = None

        self._hard_update(self.target_actor_hypernet, self.actor_hypernet)
        self._hard_update(self.target_critic, self.critic)
        self._hard_update_agent_embeddings()

    def __repr__(self):
        critic_type = (
            f'centralized/{self.centralized_critic_mode}'
            if self.centralized_critic
            else 'local'
        )
        reward_type = 'mplight' if self.use_mplight_reward else 'mean_waiting'
        return (
            f"HyperLightTD3Agent(sub_agents={self.sub_agents}, state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, actor_hypernet={self.actor_hypernet_type}, "
            f"critic_hypernet={self.critic_hypernet_type}, hyper_heads={self.hyper_head_mode}, "
            f"embedding={self.embedding_mode}, critic={critic_type}, reward={reward_type}, "
            f"discrete_td3={self.discrete_td3_mode}, "
            f"test_action={self.test_action_mode}@T={self.test_temperature:g}, device={self.device})"
        )

    def _build_actor_hypernet(self, cfg):
        return build_hypernetwork(
            self.actor_hypernet_type,
            self.meta_dim,
            self.hyper_hidden,
            self.actor_param_dim,
            dropout=self.hyper_dropout,
            target_layout=self.actor_layout,
            head_mode=self.hyper_head_mode,
            use_bias=self.hyper_use_bias,
            head_init_gain=float(cfg.get('hyper_actor_head_init_gain', self.hyper_head_init_gain)),
            **self.actor_rf_init_config,
        )

    def _build_critic(self, cfg):
        return HyperTwinTD3Critic(
            self.critic_state_dim,
            self.critic_action_dim,
            self.meta_dim,
            self.critic_hidden,
            self.critic_hyper_hidden,
            hypernet_type=self.critic_hypernet_type,
            head_mode=self.hyper_head_mode,
            use_bias=self.hyper_use_bias,
            head_init_gain=float(cfg.get('hyper_critic_head_init_gain', self.hyper_head_init_gain)),
            dropout=self.hyper_dropout,
            chunk_size=self.critic_chunk_size,
            rf_init_config=self.critic_rf_init_config,
            activation=self.activation,
        )

    def _build_generators(self):
        self.ob_generator = []
        self.reward_generator = []
        self.phase_generator = []
        self.queue_generator = []
        self.delay_generator = []

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

    def _build_action_mask(self):
        mask = torch.zeros((self.sub_agents, self.action_dim), dtype=torch.bool)
        for idx, phase_num in enumerate(self.phase_lengths):
            mask[idx, : max(1, int(phase_num))] = True
        return mask

    @staticmethod
    def _build_layout_from_module(module):
        layout = []
        offset = 0
        for name, param in module.named_parameters():
            numel = int(param.numel())
            layout.append((name, tuple(param.shape), offset, offset + numel))
            offset += numel
        return layout

    def _activate(self, x):
        if self.activation == 'tanh':
            return torch.tanh(x)
        return F.relu(x)

    def _agent_meta(self, batch_size, use_target=False):
        embeddings = self.target_agent_embeddings if use_target else self.agent_embeddings
        return embeddings.unsqueeze(0).expand(batch_size, -1, -1)

    def _actor_forward(self, state_tensor, theta):
        params = {}
        for name, shape, start, end in self.actor_layout:
            params[name] = theta[..., start:end].view(*theta.shape[:-1], *shape)

        x = torch.einsum('bni,bnoi->bno', state_tensor, params['fc1.weight']) + params['fc1.bias']
        x = self._activate(x)
        x = torch.einsum('bni,bnoi->bno', x, params['fc2.weight']) + params['fc2.bias']
        x = self._activate(x)
        return torch.einsum('bni,bnoi->bno', x, params['fc3.weight']) + params['fc3.bias']

    def _policy_logits(self, state_tensor, use_target=False):
        hypernet = self.target_actor_hypernet if use_target else self.actor_hypernet
        theta = hypernet(self._agent_meta(state_tensor.shape[0], use_target=use_target))
        logits = self._actor_forward(state_tensor, theta)
        return logits.masked_fill(~self.action_mask.unsqueeze(0), -1e9)

    def _normalize_action_probs(self, probs):
        probs = probs.clamp_min(0.0) * self.action_mask.unsqueeze(0).float()
        denom = probs.sum(dim=-1, keepdim=True)
        valid_prior = self.action_mask.float()
        valid_prior = valid_prior / valid_prior.sum(dim=-1, keepdim=True).clamp_min(1.0)
        valid_prior = valid_prior.unsqueeze(0).expand_as(probs)
        return torch.where(denom > 1e-8, probs / denom.clamp_min(1e-8), valid_prior)

    def _policy_probs(self, state_tensor, use_target=False):
        return self._normalize_action_probs(torch.softmax(self._policy_logits(state_tensor, use_target), dim=-1))

    def _critic_inputs(self, state_tensor, action_tensor):
        return self._critic_state_input(state_tensor), self._critic_action_input(action_tensor)

    def _critic_state_input(self, state_tensor):
        if not self.centralized_critic:
            return state_tensor

        if self.centralized_critic_mode == 'concat':
            state_global = state_tensor.reshape(state_tensor.shape[0], -1)
            return state_global.unsqueeze(1).expand(-1, self.sub_agents, -1)

        state_mean = state_tensor.mean(dim=1, keepdim=True)
        state_std = state_tensor.std(dim=1, unbiased=False, keepdim=True)
        state_max = state_tensor.max(dim=1, keepdim=True).values
        state_min = state_tensor.min(dim=1, keepdim=True).values
        state_context = torch.cat([state_mean, state_std, state_max, state_min], dim=-1)
        state_context = state_context.expand(-1, self.sub_agents, -1)
        return torch.cat([state_tensor, state_context], dim=-1)

    def _critic_action_input(self, action_tensor, context_action_tensor=None):
        if not self.centralized_critic:
            return action_tensor

        if self.centralized_critic_mode == 'concat':
            action_global = action_tensor.reshape(action_tensor.shape[0], -1)
            return action_global.unsqueeze(1).expand(-1, self.sub_agents, -1)

        context_action = action_tensor if context_action_tensor is None else context_action_tensor
        action_mean = context_action.mean(dim=1, keepdim=True)
        action_std = context_action.std(dim=1, unbiased=False, keepdim=True)
        action_max = context_action.max(dim=1, keepdim=True).values
        action_min = context_action.min(dim=1, keepdim=True).values
        action_context = torch.cat([action_mean, action_std, action_max, action_min], dim=-1)
        action_context = action_context.expand(-1, self.sub_agents, -1)
        return torch.cat([action_tensor, action_context], dim=-1)

    def _q_values_for_all_actions(self, critic, state_tensor, base_action_tensor, meta):
        if self.centralized_critic and self.centralized_critic_mode == 'concat':
            raise RuntimeError('expected_q mode is not supported with concat centralized critic')

        critic_state = self._critic_state_input(state_tensor)
        q1_values = []
        q2_values = []
        for action_idx in range(self.action_dim):
            candidate_action = torch.zeros_like(base_action_tensor)
            candidate_action[..., action_idx] = 1.0
            candidate_action = candidate_action * self.action_mask.unsqueeze(0).float()
            critic_action = self._critic_action_input(
                candidate_action,
                context_action_tensor=base_action_tensor,
            )
            q1, q2 = critic(critic_state, critic_action, meta, reduce=False)
            q1_values.append(q1.unsqueeze(-1))
            q2_values.append(q2.unsqueeze(-1))

        invalid_mask = ~self.action_mask.unsqueeze(0)
        return (
            torch.cat(q1_values, dim=-1).masked_fill(invalid_mask, 0.0),
            torch.cat(q2_values, dim=-1).masked_fill(invalid_mask, 0.0),
        )

    def _build_state_np(self, obs, phase):
        if self.phase:
            if self.one_hot:
                phase_feat = utils.idx2onehot(phase.astype(np.int64), self.action_dim).astype(np.float32)
            else:
                phase_feat = phase.astype(np.float32)[:, np.newaxis]
            state = np.concatenate([obs, phase_feat], axis=-1)
        else:
            state = obs
        return state.astype(np.float32)

    def _policy_prob_from_np(self, ob, phase):
        state = self._build_state_np(np.asarray(ob, dtype=np.float32), np.asarray(phase, dtype=np.int64))
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self._policy_probs(state_t).squeeze(0).cpu()

    def reset(self):
        self._build_generators()
        self._cached_action_prob = None

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
        if self.use_mplight_reward:
            rewards = []
            for reward_gen in self.reward_generator:
                rewards.append(reward_gen.generate())
            return np.asarray(np.squeeze(np.array(rewards)), dtype=np.float32).reshape(-1)

        rewards = []
        for reward_gen in self.reward_generator:
            reward = np.asarray(reward_gen.generate(), dtype=np.float32)
            rewards.append(float(np.mean(reward)))
        return np.asarray(rewards, dtype=np.float32)

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
        probs = self._policy_prob_from_np(ob, phase)
        self._cached_action_prob = probs
        probs_np = probs.numpy()

        if test:
            if self.test_action_mode == 'sample':
                return self._sample_actions_from_probs(probs_np, temperature=self.test_temperature)
            return self._greedy_actions_from_probs(probs_np)

        actions = []
        for idx in range(self.sub_agents):
            valid_dim = max(1, int(self.phase_lengths[idx]))
            if np.random.rand() < self.epsilon:
                actions.append(np.random.randint(0, valid_dim))
                continue
            actions.append(self._sample_one_from_probs(probs_np[idx, :valid_dim]))
        return np.asarray(actions, dtype=np.int64)

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
            actions.append(self._sample_one_from_probs(prob))
        return np.asarray(actions, dtype=np.int64)

    @staticmethod
    def _sample_one_from_probs(prob):
        prob = np.asarray(prob, dtype=np.float64)
        prob_sum = prob.sum()
        if prob_sum <= 1e-8 or not np.isfinite(prob_sum):
            return int(np.random.randint(0, max(1, prob.shape[0])))
        return int(np.random.choice(prob.shape[0], p=prob / prob_sum))

    def get_action_prob(self, ob, phase):
        if self._cached_action_prob is not None:
            cached = self._cached_action_prob
            self._cached_action_prob = None
            return cached
        return self._policy_prob_from_np(ob, phase)

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        state = self._build_state_np(np.asarray(last_obs, dtype=np.float32), np.asarray(last_phase, dtype=np.int64))
        next_state = self._build_state_np(np.asarray(obs, dtype=np.float32), np.asarray(cur_phase, dtype=np.int64))
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)

        if np.isscalar(done):
            done_arr = np.full((self.sub_agents,), float(done), dtype=np.float32)
        else:
            done_arr = np.asarray(done, dtype=np.float32).reshape(-1)
            if done_arr.shape[0] != self.sub_agents:
                done_arr = np.full((self.sub_agents,), float(done_arr[0]), dtype=np.float32)

        self.replay_buffer.append((state, next_state, actions, rewards, done_arr))

    def _sample_batch(self, samples):
        states, next_states, actions, rewards, dones = zip(*samples)
        return (
            torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(next_states), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(actions), dtype=torch.long, device=self.device),
            torch.tensor(np.asarray(rewards), dtype=torch.float32, device=self.device) * self.reward_scale,
            torch.tensor(np.asarray(dones), dtype=torch.float32, device=self.device),
        )

    def _huber_loss(self, prediction, target):
        beta = max(self.huber_beta, 1e-6)
        error = prediction - target
        abs_error = error.abs()
        quadratic = 0.5 * error.pow(2) / beta
        linear = abs_error - 0.5 * beta
        return torch.where(abs_error < beta, quadratic, linear).mean()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        self.train_step += 1
        samples = random.sample(self.replay_buffer, self.batch_size)
        state_t, next_state_t, action_t, reward_t, done_t = self._sample_batch(samples)
        meta = self._agent_meta(state_t.shape[0])
        target_meta = self._agent_meta(state_t.shape[0], use_target=True)

        action_onehot = F.one_hot(action_t, num_classes=self.action_dim).float()
        action_onehot = action_onehot * self.action_mask.unsqueeze(0).float()
        critic_state, critic_action = self._critic_inputs(state_t, action_onehot)
        q1_current, q2_current = self.critic(critic_state, critic_action, meta, reduce=False)

        with torch.no_grad():
            next_probs = self._policy_probs(next_state_t, use_target=True)
            if self.target_policy_noise > 0.0:
                noise = torch.randn_like(next_probs) * self.target_policy_noise
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_probs = self._normalize_action_probs(next_probs + noise)

            if self.discrete_td3_mode == 'expected_q':
                q1_target_all, q2_target_all = self._q_values_for_all_actions(
                    self.target_critic,
                    next_state_t,
                    next_probs,
                    target_meta,
                )
                min_q_target = (next_probs * torch.min(q1_target_all, q2_target_all)).sum(dim=-1)
            else:
                target_critic_state, target_critic_action = self._critic_inputs(next_state_t, next_probs)
                q1_target, q2_target = self.target_critic(
                    target_critic_state,
                    target_critic_action,
                    target_meta,
                    reduce=False,
                )
                min_q_target = torch.min(q1_target, q2_target)
            target_q = reward_t + self.gamma * (1.0 - done_t) * min_q_target

        critic_loss = self._huber_loss(q1_current, target_q) + self._huber_loss(q2_current, target_q)
        if not torch.isfinite(critic_loss):
            return 0.0

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self._critic_optimizer_parameters(), self.grad_clip)
        self.critic_optimizer.step()

        if self.train_step % self.policy_delay == 0 and self.train_step >= self.actor_warmup_steps:
            meta = self._agent_meta(state_t.shape[0])
            self._set_requires_grad(self.critic, False)
            policy_probs = self._policy_probs(state_t, use_target=False)
            if self.discrete_td3_mode == 'expected_q':
                q1_all, _ = self._q_values_for_all_actions(self.critic, state_t, policy_probs, meta)
                actor_q = (policy_probs * q1_all).sum(dim=-1).mean()
            else:
                actor_critic_state, actor_critic_action = self._critic_inputs(state_t, policy_probs)
                actor_q = self.critic.q1(actor_critic_state, actor_critic_action, meta, reduce=False).mean()
            entropy = -(policy_probs * torch.log(policy_probs.clamp_min(1e-8))).sum(dim=-1).mean()
            actor_loss = -(actor_q + self.actor_entropy_coef * entropy)

            if torch.isfinite(actor_loss):
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                clip_grad_norm_(self._actor_optimizer_parameters(), self.grad_clip)
                self.actor_optimizer.step()

                self._soft_update(self.target_actor_hypernet, self.actor_hypernet, self.tau)
                self._soft_update(self.target_critic, self.critic, self.tau)
                self._soft_update_agent_embeddings()
            self._set_requires_grad(self.critic, True)

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(critic_loss.detach().cpu().item())

    def _actor_optimizer_parameters(self):
        params = list(self.actor_hypernet.parameters())
        if isinstance(self.agent_embeddings, nn.Parameter):
            params.append(self.agent_embeddings)
        return params

    def _critic_optimizer_parameters(self):
        params = list(self.critic.parameters())
        if isinstance(self.agent_embeddings, nn.Parameter):
            params.append(self.agent_embeddings)
        return params

    def update_target_network(self):
        if not self.external_target_update:
            return
        self._soft_update(self.target_actor_hypernet, self.actor_hypernet, self.tau)
        self._soft_update(self.target_critic, self.critic, self.tau)
        self._soft_update_agent_embeddings()

    def save_model(self, e=0):
        model_dir = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        payload = {
            'actor_hypernet': self.actor_hypernet.state_dict(),
            'target_actor_hypernet': self.target_actor_hypernet.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'embedding_mode': self.embedding_mode,
            'agent_embeddings': self.agent_embeddings.detach().cpu(),
            'target_agent_embeddings': self.target_agent_embeddings.detach().cpu(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
        }
        torch.save(payload, os.path.join(model_dir, f'{e}_{self.rank}.pt'))

    def load_model(self, e=0):
        model_dir = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        checkpoint = torch.load(os.path.join(model_dir, f'{e}_{self.rank}.pt'), map_location=self.device)

        self.actor_hypernet.load_state_dict(checkpoint['actor_hypernet'])
        self.critic.load_state_dict(checkpoint['critic'])
        if isinstance(self.agent_embeddings, nn.Parameter) and 'agent_embeddings' in checkpoint:
            self.agent_embeddings.data.copy_(checkpoint['agent_embeddings'].to(self.device))
        if isinstance(self.target_agent_embeddings, nn.Parameter):
            if 'target_agent_embeddings' in checkpoint:
                self.target_agent_embeddings.data.copy_(checkpoint['target_agent_embeddings'].to(self.device))
            else:
                self._hard_update_agent_embeddings()
        if 'target_actor_hypernet' in checkpoint:
            self.target_actor_hypernet.load_state_dict(checkpoint['target_actor_hypernet'])
        else:
            self._hard_update(self.target_actor_hypernet, self.actor_hypernet)
        if 'target_critic' in checkpoint:
            self.target_critic.load_state_dict(checkpoint['target_critic'])
        else:
            self._hard_update(self.target_critic, self.critic)
        if 'actor_optimizer' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        if 'critic_optimizer' in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.epsilon = float(checkpoint.get('epsilon', self.epsilon))
        self.train_step = int(checkpoint.get('train_step', self.train_step))

    @staticmethod
    def _hard_update(target, source):
        target.load_state_dict(source.state_dict())

    @staticmethod
    def _soft_update(target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

    def _hard_update_agent_embeddings(self):
        if isinstance(self.agent_embeddings, nn.Parameter):
            self.target_agent_embeddings.data.copy_(self.agent_embeddings.data)

    def _soft_update_agent_embeddings(self):
        if isinstance(self.agent_embeddings, nn.Parameter):
            self.target_agent_embeddings.data.copy_(
                self.target_agent_embeddings.data * (1.0 - self.tau)
                + self.agent_embeddings.data * self.tau
            )

    @staticmethod
    def _set_requires_grad(module, requires_grad):
        for param in module.parameters():
            param.requires_grad_(requires_grad)


@Registry.register_model('hyper_matd3')
@Registry.register_model('hyperlight_matd3')
class HyperLightMATD3Agent(HyperLightTD3Agent):
    """
    MATD3-style registration. Use centralized_critic=True in config for the
    multi-agent critic; the actor and TSC observation/action path stay shared
    with HyperLightTD3Agent.
    """

    pass
