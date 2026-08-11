import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from .native_ppo import NativePPOAgent
from common.registry import Registry


class AgentTokenEmbedding(nn.Module):
    def __init__(self, num_agents, d_model, mode='learned'):
        super().__init__()
        self.num_agents = int(num_agents)
        self.d_model = int(d_model)
        self.mode = str(mode or 'learned').lower()
        if self.mode == 'learned':
            self.embedding = nn.Embedding(self.num_agents, self.d_model)
            nn.init.orthogonal_(self.embedding.weight)
        elif self.mode == 'one_hot':
            self.embedding = nn.Linear(self.num_agents, self.d_model, bias=False)
            nn.init.xavier_uniform_(self.embedding.weight)
            self.register_buffer('agent_eye', torch.eye(self.num_agents, dtype=torch.float32))
        elif self.mode == 'none':
            self.register_buffer('zeros', torch.zeros(self.num_agents, self.d_model))
        else:
            raise ValueError(f"Unknown MAT agent embedding mode: {self.mode}")

    def forward(self, batch_size, device):
        if self.mode == 'learned':
            ids = torch.arange(self.num_agents, dtype=torch.long, device=device)
            token = self.embedding(ids)
        elif self.mode == 'one_hot':
            token = self.embedding(self.agent_eye.to(device))
        else:
            token = self.zeros.to(device)
        return token.unsqueeze(0).expand(batch_size, -1, -1)


class MATPolicyNetwork(nn.Module):
    """
    Multi-agent transformer policy.

    The scalable default is parallel decoding with a shared start token. The
    optional autoregressive mode teacher-forces previous actions during PPO
    updates and samples one intersection at a time during acting.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        num_agents,
        d_model=128,
        n_heads=4,
        encoder_layers=2,
        decoder_layers=1,
        ffn_dim=256,
        dropout=0.1,
        activation='gelu',
        agent_embedding_mode='learned',
        decode_mode='parallel',
        norm_first=True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        self.d_model = int(d_model)
        self.decode_mode = str(decode_mode or 'parallel').lower()
        if self.decode_mode not in ('parallel', 'autoregressive'):
            raise ValueError(f"Unknown MAT decode_mode: {self.decode_mode}")

        self.state_proj = nn.Linear(self.state_dim, self.d_model)
        self.agent_tokens = AgentTokenEmbedding(
            self.num_agents,
            self.d_model,
            mode=agent_embedding_mode,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(encoder_layers))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(decoder_layers))
        self.start_action = self.action_dim
        self.action_tokens = nn.Embedding(self.action_dim + 1, self.d_model)
        self.action_head = nn.Linear(self.d_model, self.action_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.state_proj.weight)
        nn.init.zeros_(self.state_proj.bias)
        nn.init.normal_(self.action_tokens.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def _causal_mask(self, device):
        return torch.triu(
            torch.full((self.num_agents, self.num_agents), float('-inf'), device=device),
            diagonal=1,
        )

    def _shift_actions(self, actions):
        shifted = torch.full_like(actions, fill_value=self.start_action)
        if actions.shape[1] > 1:
            shifted[:, 1:] = actions[:, :-1].clamp(0, self.action_dim - 1)
        return shifted

    def encode(self, state_tensor):
        agent_tokens = self.agent_tokens(state_tensor.shape[0], state_tensor.device)
        token = self.state_proj(state_tensor) + agent_tokens
        return self.encoder(token)

    def forward(self, state_tensor, actions=None):
        memory = self.encode(state_tensor)
        agent_tokens = self.agent_tokens(state_tensor.shape[0], state_tensor.device)
        if self.decode_mode == 'autoregressive' and actions is not None:
            decoder_actions = self._shift_actions(actions)
            tgt_mask = self._causal_mask(state_tensor.device)
        else:
            decoder_actions = torch.full(
                (state_tensor.shape[0], self.num_agents),
                fill_value=self.start_action,
                dtype=torch.long,
                device=state_tensor.device,
            )
            tgt_mask = None
        target = self.action_tokens(decoder_actions) + agent_tokens
        hidden = self.decoder(target, memory, tgt_mask=tgt_mask)
        return self.action_head(hidden)


class MATValueNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        num_agents,
        d_model=128,
        n_heads=4,
        layers=2,
        ffn_dim=256,
        dropout=0.1,
        activation='gelu',
        agent_embedding_mode='learned',
        norm_first=True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_agents = int(num_agents)
        self.d_model = int(d_model)
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.agent_tokens = AgentTokenEmbedding(
            self.num_agents,
            self.d_model,
            mode=agent_embedding_mode,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(layers))
        self.value_head = nn.Linear(self.d_model, 1)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, value_input):
        agent_tokens = self.agent_tokens(value_input.shape[0], value_input.device)
        token = self.input_proj(value_input) + agent_tokens
        hidden = self.encoder(token)
        return self.value_head(hidden).squeeze(-1)


@Registry.register_model('mat')
@Registry.register_model('mat_mappo')
class MATAgent(NativePPOAgent):
    """
    MAT/MAPPO baseline for TSC.

    This keeps LibSignal's existing PPO rollout, GAE and clipping machinery,
    and replaces the shared MLP actor/critic with multi-agent transformer
    policy/value networks. Hypernetwork conditioning is intentionally excluded
    here so this class can serve as a clean MAT baseline.
    """

    def __init__(self, world, rank):
        super().__init__(world, rank)
        cfg = Registry.mapping['model_mapping']['setting'].param

        self.policy_objective = str(cfg.get('policy_objective', 'ppo')).lower()
        if self.policy_objective not in ('ppo', 'spo'):
            raise ValueError(f"Unknown MAT policy_objective: {self.policy_objective}")
        self.spo_eps = float(cfg.get('spo_eps', self.clip_eps))
        if self.spo_eps <= 0.0:
            raise ValueError('spo_eps must be positive')

        self.reward_mode = str(cfg.get('reward_mode', 'queue')).lower()
        reward_mode_aliases = {
            'mean_waiting': 'queue',
            'waiting': 'queue',
            'mplight': 'queue',
            'pressure': 'pressure_abs',
        }
        self.reward_mode = reward_mode_aliases.get(self.reward_mode, self.reward_mode)
        if self.reward_mode not in ('queue', 'pressure_abs', 'queue_pressure'):
            raise ValueError(f"Unknown MAT reward_mode: {self.reward_mode}")
        self.pressure_balance_coef = float(cfg.get('pressure_balance_coef', 0.2))
        self.pressure_release_coef = float(cfg.get('pressure_release_coef', 0.0))
        self._last_abs_pressure = None
        self.pressure_lanes = []
        self.pressure_norms = np.ones((self.sub_agents,), dtype=np.float32)
        if self._uses_pressure_reward():
            self.world.subscribe(['lane_count'])
            self.pressure_lanes, self.pressure_norms = self._build_pressure_meta()

        self.mat_d_model = int(cfg.get('mat_d_model', 128))
        self.mat_n_heads = int(cfg.get('mat_n_heads', 4))
        self.mat_encoder_layers = int(cfg.get('mat_encoder_layers', 2))
        self.mat_decoder_layers = int(cfg.get('mat_decoder_layers', 1))
        self.mat_value_layers = int(cfg.get('mat_value_layers', self.mat_encoder_layers))
        self.mat_ffn_dim = int(cfg.get('mat_ffn_dim', 4 * self.mat_d_model))
        self.mat_dropout = float(cfg.get('mat_dropout', 0.1))
        self.mat_activation = str(cfg.get('mat_activation', 'gelu')).lower()
        if self.mat_activation not in ('relu', 'gelu'):
            raise ValueError(f"Unknown MAT activation: {self.mat_activation}")
        self.mat_agent_embedding_mode = str(
            cfg.get('mat_agent_embedding_mode', cfg.get('agent_embedding_mode', 'learned'))
        ).lower()
        self.mat_decode_mode = str(cfg.get('mat_decode_mode', 'parallel')).lower()
        self.mat_norm_first = bool(cfg.get('mat_norm_first', True))

        self.actor = MATPolicyNetwork(
            self.state_dim,
            self.action_space.n,
            self.sub_agents,
            d_model=self.mat_d_model,
            n_heads=self.mat_n_heads,
            encoder_layers=self.mat_encoder_layers,
            decoder_layers=self.mat_decoder_layers,
            ffn_dim=self.mat_ffn_dim,
            dropout=self.mat_dropout,
            activation=self.mat_activation,
            agent_embedding_mode=self.mat_agent_embedding_mode,
            decode_mode=self.mat_decode_mode,
            norm_first=self.mat_norm_first,
        ).to(self.device)
        self.value = MATValueNetwork(
            self.value_input_dim,
            self.sub_agents,
            d_model=self.mat_d_model,
            n_heads=self.mat_n_heads,
            layers=self.mat_value_layers,
            ffn_dim=self.mat_ffn_dim,
            dropout=self.mat_dropout,
            activation=self.mat_activation,
            agent_embedding_mode=self.mat_agent_embedding_mode,
            norm_first=self.mat_norm_first,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self._optimizer_parameters(),
            lr=float(cfg.get('learning_rate', 3e-4)),
            eps=float(cfg.get('adam_eps', 1e-5)),
        )

    def __repr__(self):
        critic_type = (
            f'centralized/{self.centralized_critic_mode}'
            if self.centralized_critic
            else 'local'
        )
        return (
            f"MATAgent(sub_agents={self.sub_agents}, state_dim={self.state_dim}, "
            f"action_dim={self.action_space.n}, d_model={self.mat_d_model}, "
            f"heads={self.mat_n_heads}, enc_layers={self.mat_encoder_layers}, "
            f"dec_layers={self.mat_decoder_layers}, decode={self.mat_decode_mode}, "
            f"embedding={self.mat_agent_embedding_mode}, reward={self.reward_mode}, "
            f"critic={critic_type}, objective={self.policy_objective}, "
            f"test_action={self.test_action_mode}@T={self.test_temperature:g}, "
            f"device={self.device})"
        )

    def _uses_pressure_reward(self):
        return self.reward_mode in ('pressure_abs', 'queue_pressure')

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

    def _build_pressure_meta(self):
        pressure_lanes = []
        pressure_norms = []
        for inter in self.world.intersections:
            in_lanes, out_lanes = self._build_pressure_lanes(inter)
            pressure_lanes.append((in_lanes, out_lanes))
            pressure_norms.append(float(max(len(in_lanes), 1)))
        return pressure_lanes, np.asarray(pressure_norms, dtype=np.float32)

    @staticmethod
    def _lane_count_value(lane_count, lane_id):
        if isinstance(lane_count, dict):
            return float(lane_count.get(lane_id, 0.0))
        return 0.0

    def reset(self):
        super().reset()
        self._last_abs_pressure = None

    def _policy_value(self, state_tensor, action_tensor=None):
        logits = self.actor(state_tensor, action_tensor)
        logits = logits.masked_fill(~self.action_mask.unsqueeze(0), -1e9)
        values = self.value(self._value_input(state_tensor))
        return logits, values

    def _policy_prob_from_np(self, ob, phase):
        state = self._build_state_np(ob, phase)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, values = self._policy_value(state_t)
            probs = torch.softmax(logits.squeeze(0), dim=-1)
        return probs.cpu(), values.squeeze(0).cpu()

    def get_action(self, ob, phase, test=False):
        if self.mat_decode_mode != 'autoregressive':
            return super().get_action(ob, phase, test=test)

        state = self._build_state_np(np.asarray(ob, dtype=np.float32), np.asarray(phase, dtype=np.int64))
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = torch.zeros((1, self.sub_agents), dtype=torch.long, device=self.device)
        probs_out = torch.zeros((self.sub_agents, self.action_space.n), dtype=torch.float32)
        with torch.no_grad():
            for agent_idx in range(self.sub_agents):
                logits, values = self._policy_value(state_t, actions)
                probs = torch.softmax(logits[0, agent_idx], dim=-1).detach().cpu().numpy()
                valid_dim = max(1, int(self.phase_lengths[agent_idx]))
                valid_probs = probs[:valid_dim].astype(np.float64)
                if test and self.test_action_mode == 'argmax':
                    action = int(np.argmax(valid_probs))
                else:
                    temperature = self.test_temperature if test else 1.0
                    if temperature != 1.0:
                        valid_probs = np.power(np.clip(valid_probs, 1e-12, 1.0), 1.0 / temperature)
                    prob_sum = valid_probs.sum()
                    if prob_sum <= 1e-8 or not np.isfinite(prob_sum):
                        action = np.random.randint(0, valid_dim)
                    else:
                        action = int(np.random.choice(valid_dim, p=valid_probs / prob_sum))
                actions[0, agent_idx] = action
                probs_out[agent_idx] = torch.softmax(logits[0, agent_idx], dim=-1).detach().cpu()
        self._cached_action_prob = probs_out
        self._cached_value = values.squeeze(0).detach().cpu().numpy()
        return actions.squeeze(0).detach().cpu().numpy().astype(np.int64)

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

                teacher_actions = b_action if self.mat_decode_mode == 'autoregressive' else None
                logits, values = self._policy_value(b_state, teacher_actions)
                dist = Categorical(logits=logits.reshape(-1, self.action_space.n))
                new_log_prob = dist.log_prob(b_action.reshape(-1)).view_as(b_action)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_prob - b_old_log_prob)
                if self.policy_objective == 'spo':
                    policy_objective = (
                        ratio * b_advantage
                        - b_advantage.abs() * (ratio - 1.0).pow(2) / (2.0 * self.spo_eps)
                    )
                    policy_loss = -policy_objective.mean()
                else:
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
