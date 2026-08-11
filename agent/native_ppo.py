from collections import deque
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from .rl_agent import RLAgent
from .iru import IRUNetwork
from . import utils
from common.registry import Registry
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator


class SharedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super().__init__()
        dims = [int(input_dim)] + [int(dim) for dim in hidden_dims] + [int(output_dim)]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)]
        )
        self.activation = str(activation or 'relu').lower()

    def _activate(self, x):
        if self.activation == 'tanh':
            return torch.tanh(x)
        return F.relu(x)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self._activate(layer(x))
        return self.layers[-1](x)


@Registry.register_model('ppo')
@Registry.register_model('native_ppo')
@Registry.register_model('ppo_native')
class NativePPOAgent(RLAgent):
    """
    Native parameter-sharing PPO/IPPO baseline for TSC.

    This uses the same rollout, GAE, and clipped PPO objective as HyperLight PPO,
    but replaces hypernetwork-generated actor/value weights with ordinary shared
    MLPs. Set centralized_critic=True for MAPPO-style value inputs.
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
        self.normalize_advantage = bool(cfg.get('normalize_advantage', True))

        self.test_action_mode = str(cfg.get('test_action_mode', 'argmax')).lower()
        if self.test_action_mode == 'stochastic':
            self.test_action_mode = 'sample'
        if self.test_action_mode not in ('argmax', 'sample'):
            raise ValueError(f"Unknown test_action_mode: {self.test_action_mode}")
        self.test_temperature = max(1e-6, float(cfg.get('test_temperature', 1.0)))

        self.centralized_critic = bool(cfg.get('centralized_critic', False))
        self.centralized_critic_mode = str(cfg.get('centralized_critic_mode', 'pooled')).lower()
        self.activation = str(cfg.get('activation', 'relu')).lower()
        if self.activation not in ('relu', 'tanh'):
            raise ValueError(f"Unknown native PPO activation: {self.activation}")

        self.actor_arch = str(
            cfg.get('native_actor_arch', cfg.get('native_network_arch', 'mlp'))
        ).lower()
        self.value_arch = str(
            cfg.get('native_value_arch', cfg.get('native_network_arch', 'mlp'))
        ).lower()
        architecture_aliases = {
            'shared_mlp': 'mlp',
            'interpolation_recurrent_unit': 'iru',
        }
        self.actor_arch = architecture_aliases.get(self.actor_arch, self.actor_arch)
        self.value_arch = architecture_aliases.get(self.value_arch, self.value_arch)
        if self.actor_arch not in ('mlp', 'iru'):
            raise ValueError(f"Unknown native actor architecture: {self.actor_arch}")
        if self.value_arch not in ('mlp', 'iru'):
            raise ValueError(f"Unknown native value architecture: {self.value_arch}")

        self.iru_hidden_dim = int(cfg.get('iru_hidden_dim', 64))
        self.iru_actor_hidden_dim = int(cfg.get('iru_actor_hidden_dim', self.iru_hidden_dim))
        self.iru_value_hidden_dim = int(cfg.get('iru_value_hidden_dim', self.iru_hidden_dim))
        self.iru_num_blocks = int(cfg.get('iru_num_blocks', 1))
        self.iru_layer_norm = bool(cfg.get('iru_layer_norm', True))
        default_iru_steps = int(cfg.get('iru_steps', 5))
        self.iru_actor_steps = int(cfg.get('iru_actor_steps', default_iru_steps))
        self.iru_value_steps = int(cfg.get('iru_value_steps', default_iru_steps))
        self.profile_performance = bool(cfg.get('profile_performance', False))

        self.use_agent_id = bool(cfg.get('native_use_agent_id', cfg.get('use_agent_id', True)))
        self.agent_id_mode = str(
            cfg.get('native_agent_id_mode', cfg.get('agent_id_mode', 'one_hot'))
        ).lower()
        if self.agent_id_mode in ('embedding', 'learnable'):
            self.agent_id_mode = 'learned'
        if self.agent_id_mode not in ('one_hot', 'learned'):
            raise ValueError(f"Unknown native agent id mode: {self.agent_id_mode}")

        self.actor_hidden1 = int(cfg.get('actor_hidden1', 64))
        self.actor_hidden2 = int(cfg.get('actor_hidden2', 64))
        actor_hidden = cfg.get('actor_hidden', [self.actor_hidden1, self.actor_hidden2])
        if not isinstance(actor_hidden, list):
            actor_hidden = [int(actor_hidden)]
        self.actor_hidden = [int(item) for item in actor_hidden]

        value_hidden = cfg.get('value_hidden', cfg.get('critic_hidden', [64, 64]))
        if not isinstance(value_hidden, list):
            value_hidden = [int(value_hidden)]
        self.value_hidden = [int(item) for item in value_hidden]

        self._build_generators()
        self.state_dim = self.ob_length
        if self.phase:
            self.state_dim += self.action_space.n if self.one_hot else 1

        if self.use_agent_id and self.agent_id_mode == 'learned':
            self.agent_id_dim = int(
                cfg.get('native_agent_embedding_dim', cfg.get('agent_embedding_dim', 64))
            )
        else:
            self.agent_id_dim = self.sub_agents if self.use_agent_id else 0
        self.policy_input_dim = self.state_dim + self.agent_id_dim
        if not self.centralized_critic:
            self.value_input_dim = self.state_dim + self.agent_id_dim
        elif self.centralized_critic_mode == 'concat':
            self.value_input_dim = self.state_dim * self.sub_agents + self.agent_id_dim
        elif self.centralized_critic_mode == 'pooled':
            self.value_input_dim = self.state_dim * 5 + self.agent_id_dim
        else:
            raise ValueError(f"Unknown centralized_critic_mode: {self.centralized_critic_mode}")

        self.action_mask = self._build_action_mask().to(self.device)
        self.agent_id_eye = torch.eye(self.sub_agents, dtype=torch.float32, device=self.device)
        self.agent_embeddings = None
        if self.use_agent_id and self.agent_id_mode == 'learned':
            self.agent_embeddings = nn.Embedding(self.sub_agents, self.agent_id_dim).to(self.device)
            nn.init.orthogonal_(self.agent_embeddings.weight)

        if self.actor_arch == 'iru':
            self.actor = IRUNetwork(
                self.policy_input_dim,
                self.iru_actor_hidden_dim,
                self.action_space.n,
                thinking_steps=self.iru_actor_steps,
                num_blocks=self.iru_num_blocks,
                layer_norm=self.iru_layer_norm,
            ).to(self.device)
        else:
            self.actor = SharedMLP(
                self.policy_input_dim,
                self.actor_hidden,
                self.action_space.n,
                activation=self.activation,
            ).to(self.device)

        if self.value_arch == 'iru':
            self.value = IRUNetwork(
                self.value_input_dim,
                self.iru_value_hidden_dim,
                1,
                thinking_steps=self.iru_value_steps,
                num_blocks=self.iru_num_blocks,
                layer_norm=self.iru_layer_norm,
            ).to(self.device)
        else:
            self.value = SharedMLP(
                self.value_input_dim,
                self.value_hidden,
                1,
                activation=self.activation,
            ).to(self.device)

        self.optimizer = optim.Adam(
            self._optimizer_parameters(),
            lr=float(cfg.get('learning_rate', 3e-4)),
            eps=float(cfg.get('adam_eps', 1e-5)),
        )

        buffer_size = int(trainer_cfg.get('buffer_size', max(self.ppo_rollout_steps, 1)))
        self.rollout_buffer = deque(maxlen=buffer_size)
        self.replay_buffer = self.rollout_buffer
        self._transitions_since_update = 0
        self._cached_action_prob = None
        self._cached_value = None
        self._reset_performance_diagnostics()

    def __repr__(self):
        critic_type = (
            f'centralized/{self.centralized_critic_mode}'
            if self.centralized_critic
            else 'local'
        )
        return (
            f"NativePPOAgent(sub_agents={self.sub_agents}, state_dim={self.state_dim}, "
            f"action_dim={self.action_space.n}, actor={self.actor_arch}, value={self.value_arch}, "
            f"actor_hidden={self.actor_hidden}, value_hidden={self.value_hidden}, "
            f"iru={self.iru_actor_steps}/{self.iru_value_steps}x{self.iru_num_blocks}"
            f"@{self.iru_actor_hidden_dim}/{self.iru_value_hidden_dim}, "
            f"params={self._parameter_counts()['parameter_count']}, "
            f"agent_id={self.use_agent_id}/{self.agent_id_mode}, "
            f"critic={critic_type}, test_action={self.test_action_mode}@T={self.test_temperature:g}, "
            f"device={self.device})"
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

    def _synchronize_profile_device(self):
        if self.profile_performance and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _profile_start(self):
        if not self.profile_performance:
            return None
        self._synchronize_profile_device()
        return time.perf_counter()

    def _record_profile_duration(self, kind, started_at):
        if started_at is None:
            return
        self._synchronize_profile_device()
        duration = time.perf_counter() - started_at
        if kind == 'decision':
            self._decision_time_seconds += duration
            self._decision_count += 1
        elif kind == 'update':
            self._update_time_seconds += duration
            self._update_count += 1
        else:
            raise ValueError(f"Unknown performance profile kind: {kind}")

    def _reset_performance_diagnostics(self):
        self._decision_time_seconds = 0.0
        self._decision_count = 0
        self._update_time_seconds = 0.0
        self._update_count = 0
        if self.profile_performance and self.device.type == 'cuda':
            self._synchronize_profile_device()
            torch.cuda.reset_peak_memory_stats(self.device)

    def _parameter_counts(self):
        actor_count = sum(param.numel() for param in self.actor.parameters())
        value_count = sum(param.numel() for param in self.value.parameters())
        embedding_count = 0
        if self.agent_embeddings is not None:
            embedding_count = sum(param.numel() for param in self.agent_embeddings.parameters())
        return {
            'parameter_count': int(actor_count + value_count + embedding_count),
            'actor_parameter_count': int(actor_count),
            'value_parameter_count': int(value_count),
            'embedding_parameter_count': int(embedding_count),
        }

    def get_performance_diagnostics(self):
        if not self.profile_performance:
            return {}

        diagnostics = {
            key: float(value)
            for key, value in self._parameter_counts().items()
        }
        diagnostics.update(
            {
                'decision_count': float(self._decision_count),
                'decision_latency_ms_mean': (
                    1000.0 * self._decision_time_seconds / max(1, self._decision_count)
                ),
                'decision_time_ms_total': 1000.0 * self._decision_time_seconds,
                'update_count': float(self._update_count),
                'update_time_ms_mean': (
                    1000.0 * self._update_time_seconds / max(1, self._update_count)
                ),
                'update_time_ms_total': 1000.0 * self._update_time_seconds,
                'gpu_peak_memory_mb': 0.0,
                'gpu_peak_reserved_mb': 0.0,
            }
        )
        if self.device.type == 'cuda':
            diagnostics['gpu_peak_memory_mb'] = float(
                torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
            )
            diagnostics['gpu_peak_reserved_mb'] = float(
                torch.cuda.max_memory_reserved(self.device) / (1024.0 ** 2)
            )
        return diagnostics

    def _agent_id_features(self, batch_size):
        if self.agent_embeddings is not None:
            agent_idx = torch.arange(self.sub_agents, dtype=torch.long, device=self.device)
            embeddings = self.agent_embeddings(agent_idx)
            return embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return self.agent_id_eye.unsqueeze(0).expand(batch_size, -1, -1)

    def _policy_input(self, state_tensor):
        if not self.use_agent_id:
            return state_tensor
        return torch.cat([state_tensor, self._agent_id_features(state_tensor.shape[0])], dim=-1)

    def _value_input(self, state_tensor):
        if not self.centralized_critic:
            value_input = state_tensor
        elif self.centralized_critic_mode == 'concat':
            global_state = state_tensor.reshape(state_tensor.shape[0], -1)
            value_input = global_state.unsqueeze(1).expand(-1, self.sub_agents, -1)
        elif self.centralized_critic_mode == 'pooled':
            global_mean = state_tensor.mean(dim=1, keepdim=True)
            global_std = state_tensor.std(dim=1, unbiased=False, keepdim=True)
            global_max = state_tensor.max(dim=1, keepdim=True).values
            global_min = state_tensor.min(dim=1, keepdim=True).values
            global_context = torch.cat([global_mean, global_std, global_max, global_min], dim=-1)
            value_input = torch.cat(
                [state_tensor, global_context.expand(-1, self.sub_agents, -1)],
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown centralized_critic_mode: {self.centralized_critic_mode}")

        if self.use_agent_id:
            value_input = torch.cat([value_input, self._agent_id_features(state_tensor.shape[0])], dim=-1)
        return value_input

    def _policy_value(self, state_tensor):
        policy_input = self._policy_input(state_tensor)
        flat_policy_input = policy_input.reshape(-1, policy_input.shape[-1])
        logits = self.actor(flat_policy_input).view(
            state_tensor.shape[0],
            self.sub_agents,
            self.action_space.n,
        )
        logits = logits.masked_fill(~self.action_mask.unsqueeze(0), -1e9)

        value_input = self._value_input(state_tensor)
        values = self.value(value_input.reshape(-1, value_input.shape[-1])).view(
            state_tensor.shape[0],
            self.sub_agents,
        )
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
        self._reset_performance_diagnostics()

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
        profile_started_at = self._profile_start()
        probs, values = self._policy_prob_from_np(ob, phase)
        self._cached_action_prob = probs
        self._cached_value = values.numpy()
        probs_np = probs.numpy()

        if test:
            if self.test_action_mode == 'sample':
                actions = self._sample_actions_from_probs(
                    probs_np,
                    temperature=self.test_temperature,
                )
            else:
                actions = self._greedy_actions_from_probs(probs_np)
        else:
            actions = self._sample_actions_from_probs(probs_np)

        self._record_profile_duration('decision', profile_started_at)
        return actions

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

        profile_started_at = self._profile_start()

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

        mean_loss = float(np.mean(losses)) if losses else 0.0
        self._record_profile_duration('update', profile_started_at)
        return mean_loss

    def _optimizer_parameters(self):
        params = list(self.actor.parameters()) + list(self.value.parameters())
        if self.agent_embeddings is not None:
            params += list(self.agent_embeddings.parameters())
        return params

    def update_target_network(self):
        pass

    def save_model(self, e=0):
        model_dir = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        payload = {
            'actor': self.actor.state_dict(),
            'value': self.value.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'use_agent_id': self.use_agent_id,
            'agent_id_mode': self.agent_id_mode,
            'actor_arch': self.actor_arch,
            'value_arch': self.value_arch,
            'iru_actor_steps': self.iru_actor_steps,
            'iru_value_steps': self.iru_value_steps,
            'iru_num_blocks': self.iru_num_blocks,
        }
        if self.agent_embeddings is not None:
            payload['agent_embeddings'] = self.agent_embeddings.state_dict()
        torch.save(payload, os.path.join(model_dir, f'{e}_{self.rank}.pt'))

    def load_model(self, e=0):
        model_dir = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        checkpoint = torch.load(os.path.join(model_dir, f'{e}_{self.rank}.pt'), map_location=self.device)
        expected_metadata = {
            'use_agent_id': self.use_agent_id,
            'agent_id_mode': self.agent_id_mode,
            'actor_arch': self.actor_arch,
            'value_arch': self.value_arch,
            'iru_actor_steps': self.iru_actor_steps,
            'iru_value_steps': self.iru_value_steps,
            'iru_num_blocks': self.iru_num_blocks,
        }
        mismatches = {
            key: (checkpoint[key], expected)
            for key, expected in expected_metadata.items()
            if key in checkpoint and checkpoint[key] != expected
        }
        if mismatches:
            details = ', '.join(
                f'{key}=checkpoint:{actual!r}/current:{expected!r}'
                for key, (actual, expected) in mismatches.items()
            )
            raise ValueError(f'Checkpoint architecture mismatch: {details}')
        self.actor.load_state_dict(checkpoint['actor'])
        self.value.load_state_dict(checkpoint['value'])
        if self.agent_embeddings is not None and 'agent_embeddings' in checkpoint:
            self.agent_embeddings.load_state_dict(checkpoint['agent_embeddings'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])


@Registry.register_model('mappo')
@Registry.register_model('native_mappo')
@Registry.register_model('native_mappo_learned')
@Registry.register_model('mappo_native')
@Registry.register_model('mappo_iru')
class NativeMAPPOAgent(NativePPOAgent):
    """
    MAPPO registration. Behavior is controlled by config, especially
    centralized_critic=True in configs/tsc/mappo.yml or native_mappo.yml.
    """

    pass
