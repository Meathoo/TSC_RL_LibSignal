from collections import deque
import math
import os
import random

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

try:
    from torch.func import functional_call as torch_functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call as torch_functional_call

from . import RLAgent
from .actor import BaseActor
from .critic import HyperTwinCritic
from .gat_encoder import GATEncoder
from .hypernetwork import HyperNetwork
from agent import utils
from common.registry import Registry
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator


@Registry.register_model('h2tsc')
class H2TSCAgent(RLAgent):
    """
    Hypernetwork + GAT multi-agent traffic signal control (CTDE).
    """

    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])

        cfg = Registry.mapping['model_mapping']['setting'].param

        self.world = world
        self.rank = rank
        self.sub_agents = len(self.world.intersections)

        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.phase_lengths = np.array([len(inter.phases) for inter in self.world.intersections], dtype=np.int64)
        self.action_space = gym.spaces.Discrete(int(self.phase_lengths.max()))

        self.phase = cfg.get('phase', True)
        self.one_hot = cfg.get('one_hot', True)
        self.vehicle_max = float(cfg.get('vehicle_max', 1.0))
        if self.vehicle_max <= 0:
            self.vehicle_max = 1.0

        state_features = cfg.get('state_features', ['lane_count', 'lane_waiting_count'])
        self.state_features = state_features if isinstance(state_features, list) else [state_features]

        use_cuda = cfg.get('use_cuda', True)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

        self.gamma = float(cfg.get('gamma', 0.99))
        self.tau = float(cfg.get('tau', 0.01))
        self.batch_size = int(cfg.get('batch_size', 64))
        self.grad_clip = float(cfg.get('grad_clip', 5.0))
        self.policy_delay = max(1, int(cfg.get('policy_delay', 2)))
        self.reward_scale = float(cfg.get('reward_scale', 1.0))
        self.actor_warmup_steps = max(0, int(cfg.get('actor_warmup_steps', 1000)))
        self.actor_entropy_coef = float(cfg.get('actor_entropy_coef', 0.01))
        self.target_policy_noise = float(cfg.get('target_policy_noise', 0.05))
        self.target_noise_clip = float(cfg.get('target_noise_clip', 0.10))
        self.td3_clip_target = bool(cfg.get('td3_clip_target', True))
        self.huber_beta = float(cfg.get('huber_beta', 1.0))
        self.use_system_mu = bool(cfg.get('use_system_mu', True))

        self.epsilon = float(cfg.get('epsilon', 0.5))
        self.epsilon_decay = float(cfg.get('epsilon_decay', 0.9995))
        self.epsilon_min = float(cfg.get('epsilon_min', 0.05))

        gat_hidden_dim = int(cfg.get('gat_hidden_dim', 128))
        gat_heads = int(cfg.get('gat_heads', 4))
        gat_layers = int(cfg.get('gat_layers', 2))
        gat_dropout = float(cfg.get('gat_dropout', 0.0))

        pe_dim = int(cfg.get('pe_dim', 32))
        hyper_hidden = cfg.get('hyper_hidden', [256, 512])
        if not isinstance(hyper_hidden, list):
            hyper_hidden = [int(hyper_hidden)]

        self.actor_hidden1 = int(cfg.get('actor_hidden1', 128))
        self.actor_hidden2 = int(cfg.get('actor_hidden2', 64))

        critic_hidden = cfg.get('critic_hidden', [256])
        if not isinstance(critic_hidden, list):
            critic_hidden = [int(critic_hidden)]
        critic_hyper_hidden = cfg.get('critic_hyper_hidden', hyper_hidden)
        if not isinstance(critic_hyper_hidden, list):
            critic_hyper_hidden = [int(critic_hyper_hidden)]

        actor_lr = float(cfg.get('learning_rate', 3e-4))
        critic_lr = float(cfg.get('critic_lr', actor_lr))

        # Fast batched actor path is much faster than nested functional_call loops.
        self.use_functional_call = bool(cfg.get('use_functional_call', False))

        self._build_generators()

        self.state_dim = self.ob_length
        if self.phase:
            self.state_dim += self.action_space.n if self.one_hot else 1

        self.adj = self._build_adjacency_matrix().to(self.device)
        self.action_mask = self._build_action_mask().to(self.device)
        self.node_pos = self._build_node_positions().to(self.device)
        self.pos_encoding = self._build_sinusoidal_position_encoding(self.node_pos, pe_dim).to(self.device)
        self.pe_dim = self.pos_encoding.shape[-1]
        self.static_system_mu = self._build_static_system_mu().to(self.device)
        self.dynamic_system_mu_dim = 8 if self.use_system_mu else 0
        self.system_mu_dim = int(self.static_system_mu.numel() + self.dynamic_system_mu_dim)

        self.base_actor = BaseActor(
            self.state_dim,
            self.actor_hidden1,
            self.actor_hidden2,
            self.action_space.n,
        ).to(self.device)
        for param in self.base_actor.parameters():
            param.requires_grad = False

        self.actor_param_meta = self._collect_actor_param_meta()
        self.actor_param_dim = sum(item[2] for item in self.actor_param_meta)
        self.theta_layout = self._build_theta_layout()

        meta_input_dim = gat_hidden_dim + self.pe_dim + self.state_dim + self.system_mu_dim

        self.gat_encoder = GATEncoder(
            self.state_dim,
            gat_hidden_dim,
            heads=gat_heads,
            layers=gat_layers,
            dropout=gat_dropout,
        ).to(self.device)
        self.target_gat_encoder = GATEncoder(
            self.state_dim,
            gat_hidden_dim,
            heads=gat_heads,
            layers=gat_layers,
            dropout=gat_dropout,
        ).to(self.device)

        hyper_dropout = float(cfg.get('hyper_dropout', 0.0))
        self.hypernet = HyperNetwork(meta_input_dim, hyper_hidden, self.actor_param_dim, dropout=hyper_dropout).to(self.device)
        self.target_hypernet = HyperNetwork(meta_input_dim, hyper_hidden, self.actor_param_dim, dropout=hyper_dropout).to(self.device)

        self.critic = HyperTwinCritic(
            self.state_dim,
            self.action_space.n,
            self.pe_dim + self.system_mu_dim,
            hidden_dims=tuple(critic_hidden),
            hyper_hidden=tuple(critic_hyper_hidden),
            dropout=hyper_dropout,
        ).to(self.device)
        self.target_critic = HyperTwinCritic(
            self.state_dim,
            self.action_space.n,
            self.pe_dim + self.system_mu_dim,
            hidden_dims=tuple(critic_hidden),
            hyper_hidden=tuple(critic_hyper_hidden),
            dropout=hyper_dropout,
        ).to(self.device)

        self.actor_optimizer = optim.Adam(
            list(self.gat_encoder.parameters()) + list(self.hypernet.parameters()),
            lr=actor_lr,
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self._hard_update(self.target_gat_encoder, self.gat_encoder)
        self._hard_update(self.target_hypernet, self.hypernet)
        self._hard_update(self.target_critic, self.critic)

        self.train_step = 0
        self.best_epoch = 0
        self._cached_action_prob = None

    def __repr__(self):
        return (
            f"H2TSCAgent(sub_agents={self.sub_agents}, state_dim={self.state_dim}, "
            f"action_dim={self.action_space.n}, device={self.device})"
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
            valid = max(1, int(phase_num))
            mask[idx, :valid] = True
        return mask

    def _build_adjacency_matrix(self):
        adj = torch.eye(self.sub_agents, dtype=torch.float32)

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
                    edge_weight = torch.ones((edge_index.shape[1],), dtype=torch.float32)
            except Exception:
                edge_index, edge_weight = None, None

        if edge_index is not None and edge_index.numel() > 0:
            if edge_weight is None or len(edge_weight) != edge_index.shape[1]:
                edge_weight = torch.ones((edge_index.shape[1],), dtype=torch.float32)

            for e_idx in range(edge_index.shape[1]):
                src = int(edge_index[0, e_idx])
                dst = int(edge_index[1, e_idx])
                if src >= self.sub_agents or dst >= self.sub_agents:
                    continue

                w = float(edge_weight[e_idx])
                w = max(w, 1e-3)
                if w > float(adj[src, dst]):
                    adj[src, dst] = w
                if w > float(adj[dst, src]):
                    adj[dst, src] = w

        adj.fill_diagonal_(1.0)
        return adj

    def _build_node_positions(self):
        coords = None

        if hasattr(self.world, 'intersection_points'):
            try:
                coords = np.asarray(self.world.intersection_points, dtype=np.float32)
            except Exception:
                coords = None

        if coords is None or coords.shape[0] != self.sub_agents:
            coords = np.zeros((self.sub_agents, 2), dtype=np.float32)
            if self.sub_agents > 1:
                coords[:, 0] = np.linspace(0.0, 1.0, self.sub_agents, dtype=np.float32)

        coord_min = coords.min(axis=0, keepdims=True)
        coord_max = coords.max(axis=0, keepdims=True)
        denom = np.where((coord_max - coord_min) < 1e-6, 1.0, coord_max - coord_min)
        coords = (coords - coord_min) / denom

        return torch.tensor(coords, dtype=torch.float32)

    def _build_sinusoidal_position_encoding(self, coords, pe_dim):
        pe_dim = max(4, int(pe_dim))
        quarter = max(1, pe_dim // 4)

        freqs = torch.exp(
            torch.linspace(0.0, -math.log(10000.0), quarter, device=coords.device)
        )

        x_proj = coords[:, 0:1] * freqs.unsqueeze(0)
        y_proj = coords[:, 1:2] * freqs.unsqueeze(0)

        pe = torch.cat(
            [
                torch.sin(x_proj),
                torch.cos(x_proj),
                torch.sin(y_proj),
                torch.cos(y_proj),
            ],
            dim=-1,
        )

        if pe.shape[-1] < pe_dim:
            pad = torch.zeros((coords.shape[0], pe_dim - pe.shape[-1]), device=coords.device)
            pe = torch.cat([pe, pad], dim=-1)

        return pe[:, :pe_dim]

    def _build_static_system_mu(self):
        if not self.use_system_mu:
            return torch.zeros(0, dtype=torch.float32)

        n = max(1, self.sub_agents)
        off_diag = self.adj.detach().cpu().clone()
        off_diag.fill_diagonal_(0.0)
        edge_mask = off_diag > 0.0
        degree = edge_mask.float().sum(dim=-1)
        phase_counts = torch.tensor(self.phase_lengths, dtype=torch.float32)

        if n > 1:
            density = edge_mask.float().sum() / float(n * (n - 1))
            degree_scale = float(n - 1)
        else:
            density = torch.tensor(0.0)
            degree_scale = 1.0

        pos = self.node_pos.detach().cpu()
        pos_std = pos.std(dim=0, unbiased=False)

        return torch.stack(
            [
                torch.tensor(math.log1p(n) / math.log1p(256.0), dtype=torch.float32),
                density.float(),
                degree.mean() / degree_scale,
                degree.std(unbiased=False) / degree_scale,
                phase_counts.mean() / float(self.action_space.n),
                phase_counts.std(unbiased=False) / float(self.action_space.n),
                pos_std[0],
                pos_std[1],
            ]
        )

    def _system_mu_from_state(self, state_tensor):
        if not self.use_system_mu:
            return state_tensor.new_zeros((state_tensor.shape[0], 0))

        batch_size = state_tensor.shape[0]
        static_mu = self.static_system_mu.unsqueeze(0).expand(batch_size, -1)

        traffic = state_tensor[..., : self.ob_length]
        traffic_flat = traffic.reshape(batch_size, -1)
        node_load = traffic.mean(dim=-1)
        load_mean = node_load.mean(dim=-1, keepdim=True)
        load_std = node_load.std(dim=-1, unbiased=False, keepdim=True)

        hotspot_threshold = load_mean + load_std
        dynamic_mu = torch.cat(
            [
                traffic_flat.mean(dim=-1, keepdim=True),
                traffic_flat.std(dim=-1, unbiased=False, keepdim=True),
                traffic_flat.max(dim=-1, keepdim=True).values,
                traffic_flat.min(dim=-1, keepdim=True).values,
                load_mean,
                load_std,
                node_load.max(dim=-1, keepdim=True).values - node_load.min(dim=-1, keepdim=True).values,
                (node_load > hotspot_threshold).float().mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )

        return torch.cat([static_mu, dynamic_mu], dim=-1)

    def _expand_system_mu(self, state_tensor):
        system_mu = self._system_mu_from_state(state_tensor)
        return system_mu.unsqueeze(1).expand(-1, self.sub_agents, -1)

    def _collect_actor_param_meta(self):
        meta = []
        for name, param in self.base_actor.named_parameters():
            meta.append((name, tuple(param.shape), int(param.numel())))
        return meta

    def _build_theta_layout(self):
        layout = []
        offset = 0
        for name, shape, numel in self.actor_param_meta:
            layout.append((name, shape, offset, offset + numel))
            offset += numel
        return layout

    def _theta_to_actor_params(self, theta_vec):
        params = {}
        for name, shape, start, end in self.theta_layout:
            params[name] = theta_vec[start:end].view(shape)
        return params

    def _unpack_theta_batch(self, theta):
        # theta: [B, N, P]
        params = {}
        for name, shape, start, end in self.theta_layout:
            params[name] = theta[..., start:end].view(*theta.shape[:-1], *shape)
        return params

    def _batched_actor_forward(self, state_tensor, theta):
        params = self._unpack_theta_batch(theta)

        w1 = params['fc1.weight']
        b1 = params['fc1.bias']
        w2 = params['fc2.weight']
        b2 = params['fc2.bias']
        w3 = params['fc3.weight']
        b3 = params['fc3.bias']

        h1 = torch.einsum('bni,bnoi->bno', state_tensor, w1) + b1
        h1 = F.relu(h1)

        h2 = torch.einsum('bni,bnoi->bno', h1, w2) + b2
        h2 = F.relu(h2)

        logits = torch.einsum('bni,bnoi->bno', h2, w3) + b3
        return logits

    def _functional_actor_forward(self, state_tensor, theta):
        batch_logits = []
        for b_idx in range(state_tensor.shape[0]):
            node_logits = []
            for n_idx in range(self.sub_agents):
                param_dict = self._theta_to_actor_params(theta[b_idx, n_idx])
                logit = torch_functional_call(
                    self.base_actor,
                    param_dict,
                    (state_tensor[b_idx, n_idx].unsqueeze(0),),
                )
                node_logits.append(logit.squeeze(0))
            batch_logits.append(torch.stack(node_logits, dim=0))
        return torch.stack(batch_logits, dim=0)

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

    def _policy_logits(self, state_tensor, use_target=False):
        # state_tensor: [B, N, state_dim]
        gat = self.target_gat_encoder if use_target else self.gat_encoder
        hyper = self.target_hypernet if use_target else self.hypernet

        h = gat(state_tensor, self.adj)
        pe = self.pos_encoding.unsqueeze(0).expand(state_tensor.shape[0], -1, -1)
        system_mu = self._expand_system_mu(state_tensor)
        meta_input = torch.cat([h, pe, system_mu, state_tensor], dim=-1)
        theta = hyper(meta_input)

        if self.use_functional_call:
            logits = self._functional_actor_forward(state_tensor, theta)
        else:
            logits = self._batched_actor_forward(state_tensor, theta)

        logits = logits.masked_fill(~self.action_mask.unsqueeze(0), -1e9)
        return logits

    def _critic_meta_input(self, state_tensor):
        # HypeMARL parametrizes local value functions from PE(pi) and system parameter mu.
        pe = self.pos_encoding.unsqueeze(0).expand(state_tensor.shape[0], -1, -1)
        system_mu = self._expand_system_mu(state_tensor)
        return torch.cat([pe, system_mu], dim=-1)

    def _huber_loss(self, prediction, target):
        beta = max(self.huber_beta, 1e-6)
        error = prediction - target
        abs_error = error.abs()
        quadratic = 0.5 * error.pow(2) / beta
        linear = abs_error - 0.5 * beta
        return torch.where(abs_error < beta, quadratic, linear).mean()

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
        rewards = []
        for reward_gen in self.reward_generator:
            reward = np.asarray(reward_gen.generate(), dtype=np.float32)
            rewards.append(float(np.mean(reward)))
        return np.asarray(rewards, dtype=np.float32)

    def get_phase(self):
        phase = []
        for phase_gen in self.phase_generator:
            cur = np.asarray(phase_gen.generate()).reshape(-1)
            phase.append(int(cur[0]))
        phase = np.asarray(phase, dtype=np.int64)
        phase = np.minimum(np.maximum(phase, 0), self.phase_lengths - 1)
        return phase

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
            [np.random.randint(0, max(1, int(self.phase_lengths[i]))) for i in range(self.sub_agents)],
            dtype=np.int64,
        )

    def _policy_prob_from_np(self, ob, phase):
        state = self._build_state_np(ob, phase)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self._policy_logits(state_t, use_target=False).squeeze(0)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def get_action(self, ob, phase, test=False):
        probs = self._policy_prob_from_np(ob, phase)
        probs_cpu = probs.cpu()
        self._cached_action_prob = probs_cpu

        probs_np = probs_cpu.numpy()

        if test:
            return np.argmax(probs_np, axis=-1).astype(np.int64)

        actions = []
        for idx in range(self.sub_agents):
            valid_dim = max(1, int(self.phase_lengths[idx]))
            if np.random.rand() < self.epsilon:
                actions.append(np.random.randint(0, valid_dim))
                continue

            p = probs_np[idx, :valid_dim]
            p_sum = p.sum()
            if p_sum <= 1e-8 or not np.isfinite(p_sum):
                actions.append(np.random.randint(0, valid_dim))
            else:
                p = p / p_sum
                actions.append(np.random.choice(valid_dim, p=p))

        return np.asarray(actions, dtype=np.int64)

    def get_action_prob(self, ob, phase):
        if self._cached_action_prob is not None:
            cached = self._cached_action_prob
            self._cached_action_prob = None
            return cached
        return self._policy_prob_from_np(ob, phase).cpu()

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append(
            (
                np.asarray(last_obs, dtype=np.float32),
                np.asarray(last_phase, dtype=np.int64),
                np.asarray(actions, dtype=np.int64),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(obs, dtype=np.float32),
                np.asarray(cur_phase, dtype=np.int64),
                float(done),
            )
        )

    def _sample_batch(self, samples):
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []

        for sample in samples:
            obs, phase, action, reward, next_obs, next_phase, done = sample
            states.append(self._build_state_np(obs, phase))
            next_states.append(self._build_state_np(next_obs, next_phase))
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        state_t = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        next_state_t = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        action_t = torch.tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        reward_t = torch.tensor(np.asarray(rewards), dtype=torch.float32, device=self.device)
        done_t = torch.tensor(np.asarray(dones), dtype=torch.float32, device=self.device).unsqueeze(-1)

        return state_t, next_state_t, action_t, reward_t, done_t

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        self.train_step += 1
        samples = random.sample(self.replay_buffer, self.batch_size)
        state_t, next_state_t, action_t, reward_t, done_t = self._sample_batch(samples)

        batch_size = state_t.shape[0]
        critic_meta = self._critic_meta_input(state_t)

        action_onehot = F.one_hot(action_t, num_classes=self.action_space.n).float()
        action_onehot = action_onehot * self.action_mask.unsqueeze(0).float()
        reward_local = reward_t.unsqueeze(-1) * self.reward_scale

        q1_current, q2_current = self.critic(state_t, action_onehot, critic_meta, reduce=False)

        with torch.no_grad():
            next_logits = self._policy_logits(next_state_t, use_target=True)
            next_probs = torch.softmax(next_logits, dim=-1)

            if self.target_policy_noise > 0.0:
                noise = torch.randn_like(next_probs) * self.target_policy_noise
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_probs = (next_probs + noise).clamp_min(0.0)
                next_probs = next_probs * self.action_mask.unsqueeze(0).float()
                denom = next_probs.sum(dim=-1, keepdim=True)
                valid_prior = self.action_mask.float()
                valid_prior = valid_prior / valid_prior.sum(dim=-1, keepdim=True).clamp_min(1.0)
                valid_prior = valid_prior.unsqueeze(0).expand_as(next_probs)
                next_probs = torch.where(denom > 1e-8, next_probs / denom.clamp_min(1e-8), valid_prior)

            target_meta = self._critic_meta_input(next_state_t)
            q1_target, q2_target = self.target_critic(next_state_t, next_probs, target_meta, reduce=False)
            min_q_target = torch.min(q1_target, q2_target)
            if self.td3_clip_target:
                q_current_min = torch.min(q1_current, q2_current).detach().min()
                q_current_max = torch.max(q1_current, q2_current).detach().max()
                min_q_target = min_q_target.clamp(q_current_min, q_current_max)
            target_q = reward_local + self.gamma * (1.0 - done_t.unsqueeze(1)) * min_q_target

        critic_loss = (
            self._huber_loss(q1_current, target_q)
            + self._huber_loss(q2_current, target_q)
        )

        if not torch.isfinite(critic_loss):
            return 0.0

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        if self.train_step % self.policy_delay == 0 and self.train_step >= self.actor_warmup_steps:
            policy_logits = self._policy_logits(state_t, use_target=False)
            policy_probs = torch.softmax(policy_logits, dim=-1)

            actor_q = self.critic.q1(state_t, policy_probs, critic_meta, reduce=True).mean()
            entropy = -(policy_probs * torch.log(policy_probs.clamp_min(1e-8))).sum(dim=-1).mean()
            actor_loss = -(actor_q + self.actor_entropy_coef * entropy)

            if torch.isfinite(actor_loss):
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                clip_grad_norm_(
                    list(self.gat_encoder.parameters()) + list(self.hypernet.parameters()),
                    self.grad_clip,
                )
                self.actor_optimizer.step()

                self._soft_update(self.target_gat_encoder, self.gat_encoder, self.tau)
                self._soft_update(self.target_hypernet, self.hypernet, self.tau)
                self._soft_update(self.target_critic, self.critic, self.tau)

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(critic_loss.detach().cpu().item())

    def update_target_network(self):
        self._soft_update(self.target_gat_encoder, self.gat_encoder, self.tau)
        self._soft_update(self.target_hypernet, self.hypernet, self.tau)
        self._soft_update(self.target_critic, self.critic, self.tau)

    def save_model(self, e=0):
        model_dir = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        save_path = os.path.join(model_dir, f'{e}_{self.rank}.pt')
        payload = {
            'gat_encoder': self.gat_encoder.state_dict(),
            'target_gat_encoder': self.target_gat_encoder.state_dict(),
            'hypernet': self.hypernet.state_dict(),
            'target_hypernet': self.target_hypernet.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
        }
        torch.save(payload, save_path)

    def load_model(self, e=0):
        model_dir = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        load_path = os.path.join(model_dir, f'{e}_{self.rank}.pt')
        checkpoint = torch.load(load_path, map_location=self.device)

        self.gat_encoder.load_state_dict(checkpoint['gat_encoder'])
        self.hypernet.load_state_dict(checkpoint['hypernet'])
        self.critic.load_state_dict(checkpoint['critic'])

        if 'target_gat_encoder' in checkpoint:
            self.target_gat_encoder.load_state_dict(checkpoint['target_gat_encoder'])
        else:
            self._hard_update(self.target_gat_encoder, self.gat_encoder)

        if 'target_hypernet' in checkpoint:
            self.target_hypernet.load_state_dict(checkpoint['target_hypernet'])
        else:
            self._hard_update(self.target_hypernet, self.hypernet)

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



