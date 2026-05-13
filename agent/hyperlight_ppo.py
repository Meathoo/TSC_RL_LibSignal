from collections import deque
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
from .hypernetwork import build_generated_param_scaler, build_hypernetwork
from .rl_agent import RLAgent
from . import utils
from common.registry import Registry
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator


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
        self.clip_vf = cfg.get('clip_vf', 0.2)
        self.clip_vf = None if self.clip_vf is None else float(self.clip_vf)
        self.entropy_coef = float(cfg.get('entropy_coef', cfg.get('ent_coef', 0.01)))
        self.value_coef = float(cfg.get('value_coef', cfg.get('vf_coef', 0.5)))
        self.reward_scale = float(cfg.get('reward_scale', 1.0))
        self.grad_clip = float(cfg.get('grad_clip', 0.5))
        self.ppo_epochs = max(1, int(cfg.get('ppo_epochs', 4)))
        self.ppo_rollout_steps = max(1, int(cfg.get('ppo_rollout_steps', 360)))
        self.ppo_minibatch_size = max(1, int(cfg.get('ppo_minibatch_size', 2048)))
        self.value_chunk_size = int(cfg.get('value_chunk_size', 0))
        self.normalize_advantage = bool(cfg.get('normalize_advantage', True))
        self.centralized_critic = bool(cfg.get('centralized_critic', False))
        self.centralized_critic_mode = str(cfg.get('centralized_critic_mode', 'pooled')).lower()
        self.activation = str(cfg.get('activation', 'relu')).lower()
        if self.activation not in ('relu', 'tanh'):
            raise ValueError(f"Unknown HyperLight PPO activation: {self.activation}")

        self.actor_hidden1 = int(cfg.get('actor_hidden1', 64))
        self.actor_hidden2 = int(cfg.get('actor_hidden2', 64))
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
        self.actor_rf_scaler = build_generated_param_scaler(
            cfg,
            output_gain_key='hyper_rf_actor_output_gain',
            default_output_gain=0.01,
        )
        self.value_rf_scaler = build_generated_param_scaler(
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

        self._build_generators()
        self.state_dim = self.ob_length
        if self.phase:
            self.state_dim += self.action_space.n if self.one_hot else 1
        if not self.centralized_critic:
            self.value_input_dim = self.state_dim
        elif self.centralized_critic_mode == 'concat':
            self.value_input_dim = self.state_dim * self.sub_agents
        elif self.centralized_critic_mode == 'pooled':
            self.value_input_dim = self.state_dim * 5
        else:
            raise ValueError(f"Unknown centralized_critic_mode: {self.centralized_critic_mode}")
        self.action_mask = self._build_action_mask().to(self.device)

        self.embedding_mode = str(cfg.get('agent_embedding_mode', 'one_hot')).lower()
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

        self.base_actor = BaseActor(
            self.state_dim,
            self.actor_hidden1,
            self.actor_hidden2,
            self.action_space.n,
        ).to(self.device)
        for param in self.base_actor.parameters():
            param.requires_grad = False

        self.actor_layout = self._build_layout_from_module(self.base_actor)
        self.actor_param_dim = self.actor_layout[-1][-1]
        self.actor_hypernet = build_hypernetwork(
            self.hypernet_type,
            self.meta_dim,
            hyper_hidden,
            self.actor_param_dim,
            dropout=hyper_dropout,
            target_layout=self.actor_layout,
            head_mode=self.hyper_head_mode,
            use_bias=self.hyper_use_bias,
            head_init_gain=float(cfg.get('hyper_actor_head_init_gain', self.hyper_head_init_gain)),
        ).to(self.device)

        self.value_dims = [self.value_input_dim] + self.value_hidden + [1]
        self.value_layout = self._build_layout_from_dims(self.value_dims)
        self.value_param_dim = self.value_layout[-1][-1]
        self.value_hypernet = build_hypernetwork(
            self.value_hypernet_type,
            self.meta_dim,
            value_hyper_hidden,
            self.value_param_dim,
            dropout=hyper_dropout,
            target_layout=self.value_layout,
            head_mode=self.hyper_head_mode,
            use_bias=self.hyper_use_bias,
            head_init_gain=float(cfg.get('hyper_value_head_init_gain', self.hyper_head_init_gain)),
        ).to(self.device)

        optimizer_params = list(self.actor_hypernet.parameters()) + list(self.value_hypernet.parameters())
        if isinstance(self.agent_embeddings, nn.Parameter):
            optimizer_params.append(self.agent_embeddings)
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

    def __repr__(self):
        critic_type = (
            f'centralized/{self.centralized_critic_mode}'
            if self.centralized_critic
            else 'local'
        )
        return (
            f"HyperLightPPOAgent(sub_agents={self.sub_agents}, state_dim={self.state_dim}, "
            f"action_dim={self.action_space.n}, actor_hypernet={self.hypernet_type}, "
            f"value_hypernet={self.value_hypernet_type}, hyper_heads={self.hyper_head_mode}, "
            f"embedding={self.embedding_mode}, activation={self.activation}, "
            f"value_chunk_size={self.value_chunk_size}, "
            f"critic={critic_type}, device={self.device})"
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
        mask = torch.zeros((self.sub_agents, self.action_space.n), dtype=torch.bool)
        for idx, phase_num in enumerate(self.phase_lengths):
            mask[idx, : max(1, int(phase_num))] = True
        return mask

    def _build_layout_from_module(self, module):
        layout = []
        offset = 0
        for name, param in module.named_parameters():
            numel = int(param.numel())
            layout.append((name, tuple(param.shape), offset, offset + numel))
            offset += numel
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

    def _activate(self, x):
        if self.activation == 'tanh':
            return torch.tanh(x)
        return F.relu(x)

    def _agent_meta(self, batch_size):
        return self.agent_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

    def _actor_forward(self, state_tensor, theta):
        params = {}
        for name, shape, start, end in self.actor_layout:
            params[name] = theta[..., start:end].view(*theta.shape[:-1], *shape)

        w1 = self.actor_rf_scaler.scale_weight(params['fc1.weight'], self.actor_hidden1, self.state_dim, 0, 3)
        b1 = self.actor_rf_scaler.scale_bias(params['fc1.bias'], self.state_dim, 0, 3)
        w2 = self.actor_rf_scaler.scale_weight(params['fc2.weight'], self.actor_hidden2, self.actor_hidden1, 1, 3)
        b2 = self.actor_rf_scaler.scale_bias(params['fc2.bias'], self.actor_hidden1, 1, 3)
        w3 = self.actor_rf_scaler.scale_weight(params['fc3.weight'], self.action_space.n, self.actor_hidden2, 2, 3)
        b3 = self.actor_rf_scaler.scale_bias(params['fc3.bias'], self.actor_hidden2, 2, 3)

        x = torch.einsum('bni,bnoi->bno', state_tensor, w1) + b1
        x = self._activate(x)
        x = torch.einsum('bni,bnoi->bno', x, w2) + b2
        x = self._activate(x)
        return torch.einsum('bni,bnoi->bno', x, w3) + b3

    def _generated_value_forward(self, value_input, theta):
        x = value_input
        layer_count = len(self.value_layout)
        for layer_idx, (_, _, weight_start, bias_start, end) in enumerate(self.value_layout):
            out_dim, in_dim = self.value_layout[layer_idx][1]
            weight = theta[..., weight_start:bias_start].view(*theta.shape[:-1], out_dim, in_dim)
            bias = theta[..., bias_start:end].view(*theta.shape[:-1], out_dim)
            weight = self.value_rf_scaler.scale_weight(weight, out_dim, in_dim, layer_idx, layer_count)
            bias = self.value_rf_scaler.scale_bias(bias, in_dim, layer_idx, layer_count)
            x = torch.einsum('bni,bnoi->bno', x, weight) + bias
            if layer_idx < len(self.value_layout) - 1:
                x = self._activate(x)
        return x.squeeze(-1)

    def _value_forward_from_meta(self, value_input, meta):
        chunk_size = self.value_chunk_size
        if chunk_size <= 0 or meta.shape[1] <= chunk_size:
            value_theta = self.value_hypernet(meta)
            return self._generated_value_forward(value_input, value_theta)

        values = []
        for start in range(0, meta.shape[1], chunk_size):
            end = min(start + chunk_size, meta.shape[1])
            value_theta = self.value_hypernet(meta[:, start:end])
            values.append(
                self._generated_value_forward(
                    value_input[:, start:end],
                    value_theta,
                )
            )
        return torch.cat(values, dim=1)

    def _value_input(self, state_tensor):
        if not self.centralized_critic:
            return state_tensor
        if self.centralized_critic_mode == 'concat':
            global_state = state_tensor.reshape(state_tensor.shape[0], -1)
            return global_state.unsqueeze(1).expand(-1, self.sub_agents, -1)

        global_mean = state_tensor.mean(dim=1, keepdim=True)
        global_std = state_tensor.std(dim=1, unbiased=False, keepdim=True)
        global_max = state_tensor.max(dim=1, keepdim=True).values
        global_min = state_tensor.min(dim=1, keepdim=True).values
        global_context = torch.cat([global_mean, global_std, global_max, global_min], dim=-1)
        global_context = global_context.expand(-1, self.sub_agents, -1)
        return torch.cat([state_tensor, global_context], dim=-1)

    def _policy_value(self, state_tensor):
        meta = self._agent_meta(state_tensor.shape[0])
        actor_theta = self.actor_hypernet(meta)
        logits = self._actor_forward(state_tensor, actor_theta)
        logits = logits.masked_fill(~self.action_mask.unsqueeze(0), -1e9)

        values = self._value_forward_from_meta(self._value_input(state_tensor), meta)
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

    def _policy_prob_from_np(self, ob, phase):
        state = self._build_state_np(ob, phase)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, values = self._policy_value(state_t)
            probs = torch.softmax(logits.squeeze(0), dim=-1)
        return probs.cpu(), values.squeeze(0).cpu()

    def reset(self):
        self._build_generators()
        self._cached_action_prob = None
        self._cached_value = None

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
        probs, values = self._policy_prob_from_np(ob, phase)
        self._cached_action_prob = probs
        self._cached_value = values.numpy()
        probs_np = probs.numpy()

        if test:
            return np.argmax(probs_np, axis=-1).astype(np.int64)

        actions = []
        for idx in range(self.sub_agents):
            valid_dim = max(1, int(self.phase_lengths[idx]))
            prob = probs_np[idx, :valid_dim]
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
        probs, values = self._policy_prob_from_np(ob, phase)
        self._cached_value = values.numpy()
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
            _, values = self._policy_prob_from_np(last_obs, last_phase)
            old_value = values.numpy().astype(np.float32)
        else:
            old_value = np.asarray(self._cached_value, dtype=np.float32)
        self._cached_value = None

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
            )
        )
        self._transitions_since_update += 1

    def _rollout_tensors(self, rollout):
        states, next_states, actions, rewards, dones, old_log_probs, old_values = zip(*rollout)
        return (
            torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(next_states), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(actions), dtype=torch.long, device=self.device),
            torch.tensor(np.asarray(rewards), dtype=torch.float32, device=self.device) * self.reward_scale,
            torch.tensor(np.asarray(dones), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(old_log_probs), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(old_values), dtype=torch.float32, device=self.device),
        )

    def _compute_gae(self, rewards, dones, old_values, next_states):
        with torch.no_grad():
            _, last_value = self._policy_value(next_states[-1:].detach())
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

    def train(self):
        if self._transitions_since_update < self.ppo_rollout_steps:
            return 0.0

        rollout = list(self.rollout_buffer)
        self.rollout_buffer.clear()
        self._transitions_since_update = 0

        state_t, next_state_t, action_t, reward_t, done_t, old_log_prob_t, old_value_t = self._rollout_tensors(rollout)
        advantages_t, returns_t = self._compute_gae(reward_t, done_t, old_value_t, next_state_t)

        num_steps = state_t.shape[0]
        step_batch_size = max(1, min(num_steps, self.ppo_minibatch_size // max(1, self.sub_agents)))
        losses = []

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

                logits, values = self._policy_value(b_state)
                dist = Categorical(logits=logits.reshape(-1, self.action_space.n))
                new_log_prob = dist.log_prob(b_action.reshape(-1)).view_as(b_action)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_prob - b_old_log_prob)
                policy_loss_1 = ratio * b_advantage
                policy_loss_2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantage
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

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
                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self._optimizer_parameters(), self.grad_clip)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

        return float(np.mean(losses)) if losses else 0.0

    def _optimizer_parameters(self):
        params = list(self.actor_hypernet.parameters()) + list(self.value_hypernet.parameters())
        if isinstance(self.agent_embeddings, nn.Parameter):
            params.append(self.agent_embeddings)
        return params

    def update_target_network(self):
        pass

    def save_model(self, e=0):
        model_dir = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        payload = {
            'actor_hypernet': self.actor_hypernet.state_dict(),
            'value_hypernet': self.value_hypernet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'embedding_mode': self.embedding_mode,
            'agent_embeddings': self.agent_embeddings.detach().cpu(),
        }
        torch.save(payload, os.path.join(model_dir, f'{e}_{self.rank}.pt'))

    def load_model(self, e=0):
        model_dir = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        checkpoint = torch.load(os.path.join(model_dir, f'{e}_{self.rank}.pt'), map_location=self.device)
        self.actor_hypernet.load_state_dict(checkpoint['actor_hypernet'])
        self.value_hypernet.load_state_dict(checkpoint['value_hypernet'])
        if isinstance(self.agent_embeddings, nn.Parameter) and 'agent_embeddings' in checkpoint:
            self.agent_embeddings.data.copy_(checkpoint['agent_embeddings'].to(self.device))
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])


@Registry.register_model('hyperlight_mappo')
class HyperLightMAPPOAgent(HyperLightPPOAgent):
    """
    MAPPO-style registration. The behavior is controlled by config, especially
    centralized_critic=True in configs/tsc/hyperlight_mappo.yml.
    """

    pass
