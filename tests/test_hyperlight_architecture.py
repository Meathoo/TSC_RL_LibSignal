import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from agent.actor import BaseActor
from agent.hyperlight_architecture import (
    DirectedGraphAttentionLayer,
    DirectedGraphCritic,
    MovementTokenEncoder,
)
from agent.hyperlight_ppo import HyperLightGraphMAPPOAgent, HyperLightPPOAgent
from agent.iru import IRUNetwork
from common.registry import Registry
from common.utils import load_config
from world.world_cityflow import World


class MovementTokenEncoderTests(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(4)
        dynamic = torch.randn(2, 3, 5, 2)
        mask = torch.tensor(
            [
                [True, False, False, False, False],
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        )
        phase_mask = torch.randint(0, 2, (3, 5, 4), dtype=torch.float32)
        phase = torch.nn.functional.one_hot(
            torch.tensor([[0, 1, 2], [1, 2, 3]]),
            num_classes=4,
        ).float()
        source_position = torch.linspace(0.0, 1.0, 5).repeat(3, 1)
        movement_position = torch.flip(source_position, dims=[-1])
        return dynamic, mask, phase_mask, phase, source_position, movement_position

    def test_shape_padding_and_permutation_invariance(self):
        encoder = MovementTokenEncoder(
            2,
            4,
            token_dim=16,
            output_dim=12,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        ).eval()
        inputs = self._inputs()
        output = encoder(*inputs)
        self.assertEqual((2, 3, 12), tuple(output.shape))
        self.assertTrue(torch.isfinite(output).all())

        dynamic, mask, phase_mask, phase, source_position, movement_position = inputs
        corrupted = dynamic.clone()
        corrupted.masked_fill_(~mask.unsqueeze(0).unsqueeze(-1), 1e6)
        torch.testing.assert_close(
            output,
            encoder(
                corrupted,
                mask,
                phase_mask,
                phase,
                source_position,
                movement_position,
            ),
        )

        permutation = torch.tensor([3, 0, 4, 1, 2])
        torch.testing.assert_close(
            output,
            encoder(
                dynamic[:, :, permutation],
                mask[:, permutation],
                phase_mask[:, permutation],
                phase,
                source_position[:, permutation],
                movement_position[:, permutation],
            ),
            atol=2e-6,
            rtol=2e-6,
        )

    def test_masked_tokens_have_zero_gradient(self):
        encoder = MovementTokenEncoder(
            2,
            4,
            token_dim=16,
            output_dim=8,
            num_heads=4,
            num_layers=1,
        )
        dynamic, mask, phase_mask, phase, source_position, movement_position = self._inputs()
        dynamic.requires_grad_(True)
        output = encoder(
            dynamic,
            mask,
            phase_mask,
            phase,
            source_position,
            movement_position,
        )
        weights = torch.linspace(0.1, 0.8, output.shape[-1])
        (output * weights).sum().backward()
        invalid_grad = dynamic.grad.masked_select(~mask.unsqueeze(0).unsqueeze(-1))
        valid_grad = dynamic.grad.masked_select(mask.unsqueeze(0).unsqueeze(-1))
        self.assertEqual(0.0, float(invalid_grad.abs().sum()))
        self.assertGreater(float(valid_grad.abs().sum()), 0.0)


class DirectedGraphCriticTests(unittest.TestCase):
    def test_edge_direction_is_not_symmetrized(self):
        torch.manual_seed(8)
        layer = DirectedGraphAttentionLayer(8, num_heads=2, dropout=0.0).eval()
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # 0 -> 1
        features = torch.randn(1, 3, 8)
        baseline = layer(features, edge_index)

        changed_source = features.clone()
        changed_source[:, 0] += torch.tensor([2.0, -1.0, 0.5, 3.0, -2.0, 1.0, 0.0, 4.0])
        source_output = layer(changed_source, edge_index)
        self.assertGreater(float((baseline[:, 1] - source_output[:, 1]).abs().max()), 1e-6)
        torch.testing.assert_close(baseline[:, 2], source_output[:, 2])

        changed_destination = features.clone()
        changed_destination[:, 1] += torch.randn(8) * 3.0
        destination_output = layer(changed_destination, edge_index)
        # There is no 1 -> 0 edge, so node 0 only sees its self loop.
        torch.testing.assert_close(baseline[:, 0], destination_output[:, 0])

    def test_graph_critic_shape_film_identity_and_backward(self):
        torch.manual_seed(9)
        critic = DirectedGraphCritic(
            6,
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            global_pool=True,
            meta_dim=5,
            film_hidden_dims=(8,),
            film_scale=0.1,
            film_zero_init=True,
        )
        node_features = torch.randn(2, 4, 6, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        meta_a = torch.zeros(2, 4, 5)
        meta_b = torch.randn(2, 4, 5)
        values_a = critic(node_features, edge_index, meta=meta_a)
        values_b = critic(node_features, edge_index, meta=meta_b)
        self.assertEqual((2, 4), tuple(values_a.shape))
        self.assertTrue(torch.isfinite(values_a).all())
        # Zero-initialized FiLM is exactly the identity for every meta vector.
        torch.testing.assert_close(values_a, values_b)
        values_b.mean().backward()
        self.assertGreater(float(node_features.grad.abs().sum()), 0.0)
        film_grads = [param.grad for param in critic.film.parameters()]
        self.assertTrue(any(grad is not None and grad.abs().sum() > 0 for grad in film_grads))


class FiLMActorTests(unittest.TestCase):
    def test_zero_film_matches_shared_actor_without_dense_theta(self):
        torch.manual_seed(12)
        agent = HyperLightPPOAgent.__new__(HyperLightPPOAgent)
        agent.actor_hidden1 = 7
        agent.actor_hidden2 = 5
        agent.actor_arch = 'mlp'
        agent.actor_film_param_dim = 2 * (7 + 5)
        agent.hyper_film_scale = 0.1
        agent.activation = 'relu'
        agent.base_actor = BaseActor(6, 7, 5, 3)
        state = torch.randn(2, 4, 6)
        zero_film = torch.zeros(2, 4, agent.actor_film_param_dim)
        torch.testing.assert_close(
            agent._actor_film_forward(state, zero_film),
            agent.base_actor(state),
        )
        changed = zero_film.clone()
        changed[..., :7] = 1.0
        self.assertFalse(
            torch.allclose(
                agent._actor_film_forward(state, changed),
                agent.base_actor(state),
            )
        )

    def test_zero_film_matches_shared_iru_and_updates_actor_and_adapter(self):
        torch.manual_seed(13)
        agent = HyperLightPPOAgent.__new__(HyperLightPPOAgent)
        agent.actor_arch = 'iru'
        agent.iru_actor_hidden_dim = 7
        agent.actor_film_param_dim = 4 * agent.iru_actor_hidden_dim
        agent.hyper_film_scale = 0.1
        agent.base_actor = IRUNetwork(
            input_dim=6,
            hidden_dim=agent.iru_actor_hidden_dim,
            output_dim=3,
            thinking_steps=2,
            num_blocks=1,
        )
        state = torch.randn(2, 4, 6)
        zero_film = torch.zeros(
            2,
            4,
            agent.actor_film_param_dim,
            requires_grad=True,
        )
        shared_logits = agent.base_actor(state)
        film_logits = agent._actor_film_forward(state, zero_film)
        torch.testing.assert_close(film_logits, shared_logits)

        film_logits.mean().backward()
        self.assertGreater(float(zero_film.grad.abs().sum()), 0.0)
        actor_grads = [param.grad for param in agent.base_actor.parameters()]
        self.assertTrue(
            any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in actor_grads)
        )

        changed = zero_film.detach().clone()
        changed[..., :agent.iru_actor_hidden_dim] = 1.0
        self.assertFalse(
            torch.allclose(
                agent._actor_film_forward(state, changed),
                shared_logits,
            )
        )


class HeadResidualFastPathTests(unittest.TestCase):
    @staticmethod
    def _agent():
        agent = HyperLightPPOAgent.__new__(HyperLightPPOAgent)
        agent.activation = 'relu'
        agent.actor_arch = 'mlp'
        agent.hyper_residual_actor_scale = 0.02
        agent.hyper_residual_value_scale = 0.02
        return agent

    def test_actor_fast_path_matches_materialized_theta_and_gradients(self):
        torch.manual_seed(14)
        agent = self._agent()
        agent.base_actor = BaseActor(6, 7, 5, 3)
        agent.actor_layout = agent._build_layout_from_module(agent.base_actor)
        agent.actor_head_layout, agent.actor_head_param_dim = agent._build_head_layout(
            agent.actor_layout
        )
        state = torch.randn(2, 4, 6, requires_grad=True)
        head_theta = torch.randn(
            2,
            4,
            agent.actor_head_param_dim,
            requires_grad=True,
        )

        fast, _ = agent._actor_head_residual_forward(state, head_theta)
        materialized, _, _ = agent._compose_head_theta(
            head_theta,
            agent.base_actor,
            agent.actor_head_layout,
            agent.hyper_residual_actor_scale,
        )
        reference = agent._actor_forward(state, materialized)
        torch.testing.assert_close(fast, reference, atol=2e-6, rtol=2e-6)

        parameters = [state, head_theta, *agent.base_actor.parameters()]
        fast_grads = torch.autograd.grad(fast.sum(), parameters, retain_graph=True)
        reference_grads = torch.autograd.grad(reference.sum(), parameters)
        for fast_grad, reference_grad in zip(fast_grads, reference_grads):
            torch.testing.assert_close(fast_grad, reference_grad, atol=2e-6, rtol=2e-6)

    def test_value_fast_path_matches_materialized_theta_and_diagnostics(self):
        torch.manual_seed(15)
        agent = self._agent()
        # The optimized path must preserve the generated-network activation,
        # rather than blindly executing the ReLUs stored in base_value.
        agent.activation = 'tanh'
        agent.base_value = torch.nn.Sequential(
            torch.nn.Linear(9, 7),
            torch.nn.ReLU(),
            torch.nn.Linear(7, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1),
        )
        agent.value_layout = agent._build_layout_from_dims([9, 7, 5, 1])
        agent.value_head_layout, agent.value_head_param_dim = agent._build_head_layout(
            agent.value_layout
        )
        value_input = torch.randn(2, 4, 9, requires_grad=True)
        head_theta = torch.randn(
            2,
            4,
            agent.value_head_param_dim,
            requires_grad=True,
        )

        fast, scaled_delta = agent._value_head_residual_forward(value_input, head_theta)
        materialized, base, delta = agent._compose_head_theta(
            head_theta,
            agent.base_value,
            agent.value_head_layout,
            agent.hyper_residual_value_scale,
        )
        reference = agent._generated_value_forward(value_input, materialized)
        torch.testing.assert_close(fast, reference, atol=2e-6, rtol=2e-6)

        compact_diagnostics = agent._compact_head_diagnostics(
            'value',
            agent.base_value,
            scaled_delta,
        )
        reference_diagnostics = agent._theta_diagnostics('value', base, delta, materialized)
        reference_diagnostics.update(
            agent._head_diagnostics(
                'value_head',
                base,
                delta,
                materialized,
                agent.value_head_layout,
            )
        )
        self.assertEqual(set(reference_diagnostics), set(compact_diagnostics))
        for key, expected in reference_diagnostics.items():
            self.assertAlmostEqual(expected, compact_diagnostics[key], places=5)

        parameters = [value_input, head_theta, *agent.base_value.parameters()]
        fast_grads = torch.autograd.grad(fast.sum(), parameters, retain_graph=True)
        reference_grads = torch.autograd.grad(reference.sum(), parameters)
        for fast_grad, reference_grad in zip(fast_grads, reference_grads):
            torch.testing.assert_close(fast_grad, reference_grad, atol=2e-6, rtol=2e-6)


class ParameterMatchedActorTests(unittest.TestCase):
    def test_shared_mlp_124_116_exactly_matches_iru_n1_parameter_count(self):
        shared_mlp = BaseActor(32, 124, 116, 8)
        shared_iru = IRUNetwork(
            input_dim=32,
            hidden_dim=64,
            output_dim=8,
            thinking_steps=1,
            num_blocks=1,
            layer_norm=True,
        )
        mlp_parameters = sum(param.numel() for param in shared_mlp.parameters())
        iru_parameters = sum(param.numel() for param in shared_iru.parameters())
        self.assertEqual(19_528, mlp_parameters)
        self.assertEqual(mlp_parameters, iru_parameters)


def _fake_cityflow_world():
    world = World.__new__(World)
    world.RIGHT = True
    world.subscribe = lambda fns: None
    world.intersection_ids = ['n0', 'n1', 'n2']
    world.intersection_points = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    world.roadnet = {'roads': [], 'intersections': []}
    intersections = []
    lane_count = {}
    lane_waiting = {}
    lane_delay = {}

    for node_idx, (lane_total, phase_total) in enumerate(((1, 2), (2, 3), (3, 4))):
        inter_id = f'n{node_idx}'
        in_road = {
            'id': f'in{node_idx}',
            'startIntersection': f'virtual_in{node_idx}',
            'endIntersection': inter_id,
            'lanes': [{} for _ in range(lane_total)],
        }
        out_road = {
            'id': f'out{node_idx}',
            'startIntersection': inter_id,
            'endIntersection': f'virtual_out{node_idx}',
            'lanes': [{} for _ in range(max(2, lane_total))],
        }
        incoming_lanes = [f'in{node_idx}_{idx}' for idx in range(lane_total)]
        movements = []
        for lane_idx, lane_id in enumerate(incoming_lanes):
            movements.append((lane_id, f'out{node_idx}_{lane_idx}'))
        if node_idx > 0:
            movements.append((incoming_lanes[0], f'out{node_idx}_1'))

        phase_links = [[] for _ in range(phase_total)]
        for movement_idx, movement in enumerate(movements):
            phase_links[movement_idx % phase_total].append(movement)
        phase_start_lanes = [
            sorted({movement[0] for movement in links})
            for links in phase_links
        ]
        exposed_movements = movements
        if node_idx == 2:
            # SUMO getControlledLinks may group multiple connections under one
            # signal index; the movement spec must flatten every connection.
            exposed_movements = [tuple(movements[:2]), tuple(movements[2:])]
        intersections.append(
            SimpleNamespace(
                id=inter_id,
                phases=list(range(phase_total)),
                current_phase=node_idx % phase_total,
                in_roads=[in_road],
                out_roads=[out_road],
                lanelinks=exposed_movements,
                phase_available_lanelinks=phase_links,
                phase_available_startlanes=phase_start_lanes,
                startlanes=incoming_lanes,
            )
        )
        for lane_idx, lane_id in enumerate(incoming_lanes):
            lane_count[lane_id] = float(node_idx + lane_idx + 1)
            lane_waiting[lane_id] = float(lane_idx)
            lane_delay[lane_id] = 0.2

    world.intersections = intersections
    world.id2intersection = {inter.id: inter for inter in intersections}

    def get_info(feature):
        return {
            'lane_count': lane_count,
            'lane_waiting_count': lane_waiting,
            'lane_delay': lane_delay,
        }[feature]

    world.get_info = get_info
    world.get_adjacency = lambda: (
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([1.0 / 300.0, 1.0 / 200.0]),
    )
    return world


class HyperLightGraphIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.old_model_setting = Registry.mapping['model_mapping'].get('setting')
        self.old_trainer_setting = Registry.mapping['trainer_mapping'].get('setting')
        self.old_logger_path = Registry.mapping['logger_mapping'].get('path')
        config, _, _ = load_config('configs/tsc/hyperlight_graph_mappo.yml')
        config['model']['use_cuda'] = False
        config['model']['movement_encoder_dropout'] = 0.0
        config['model']['graph_critic_dropout'] = 0.0
        Registry.mapping['model_mapping']['setting'] = SimpleNamespace(param=config['model'])
        Registry.mapping['trainer_mapping']['setting'] = SimpleNamespace(param=config['trainer'])
        self.world = _fake_cityflow_world()

    def tearDown(self):
        if self.old_model_setting is None:
            Registry.mapping['model_mapping'].pop('setting', None)
        else:
            Registry.mapping['model_mapping']['setting'] = self.old_model_setting
        if self.old_trainer_setting is None:
            Registry.mapping['trainer_mapping'].pop('setting', None)
        else:
            Registry.mapping['trainer_mapping']['setting'] = self.old_trainer_setting
        if self.old_logger_path is None:
            Registry.mapping['logger_mapping'].pop('path', None)
        else:
            Registry.mapping['logger_mapping']['path'] = self.old_logger_path

    @staticmethod
    def _state(agent):
        state = agent._build_state_np(agent.get_ob(), agent.get_phase())
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def test_policy_value_uses_raw_rollout_state_and_all_new_modules(self):
        agent = HyperLightGraphMAPPOAgent(self.world, 0)
        state = self._state(agent)
        logits, values = agent._policy_value(state)

        self.assertEqual((1, 3, 4), tuple(logits.shape))
        self.assertEqual((1, 3), tuple(values.shape))
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue((logits[0, 0, 2:] == -1e9).all())
        self.assertIsNone(agent.value_hypernet)
        self.assertEqual([[1, 2], [0, 1]], agent.graph_edge_index.tolist())
        self.assertEqual([1, 3, 4], agent.movement_token_mask.sum(dim=1).tolist())
        self.assertEqual(
            [1.0, 0.0, 0.0, 0.0],
            agent.movement_phase_availability[1, 0].tolist(),
        )
        self.assertEqual(
            [0.0, 0.0, 1.0, 0.0],
            agent.movement_phase_availability[1, 2].tolist(),
        )
        self.assertTrue(
            torch.equal(
                agent.movement_turn_features.sum(dim=-1).bool(),
                agent.movement_token_mask,
            )
        )
        self.assertLess(agent.actor_film_param_dim, agent.actor_param_dim)

        parameters = agent._optimizer_parameters()
        self.assertEqual(len(parameters), len({id(param) for param in parameters}))
        valid_logits = logits.masked_select(agent.action_mask.unsqueeze(0))
        (valid_logits.mean() + values.mean()).backward()
        for module in (
            agent.movement_encoder,
            agent.graph_critic,
            agent.actor_hypernet,
            agent.base_actor,
        ):
            gradients = [param.grad for param in module.parameters() if param.requires_grad]
            self.assertTrue(
                any(
                    grad is not None
                    and torch.isfinite(grad).all()
                    and float(grad.abs().sum()) > 0.0
                    for grad in gradients
                )
            )

    def test_shared_and_film_iru_actor_paths_keep_the_same_critic(self):
        original_cfg = Registry.mapping['model_mapping']['setting'].param
        signatures = {}
        try:
            for adapter_mode in ('none', 'film'):
                cfg = dict(original_cfg)
                cfg.update(
                    {
                        'hyper_actor_arch': 'iru',
                        'hyper_adapter_mode': adapter_mode,
                        'hyper_residual': False,
                        'iru_actor_hidden_dim': 64,
                        'iru_actor_steps': 1,
                        'iru_num_blocks': 1,
                    }
                )
                Registry.mapping['model_mapping']['setting'] = SimpleNamespace(param=cfg)
                agent = HyperLightGraphMAPPOAgent(self.world, 0)
                logits, values = agent._policy_value(self._state(agent))
                self.assertEqual((1, 3, 4), tuple(logits.shape))
                self.assertEqual((1, 3), tuple(values.shape))
                self.assertIsInstance(agent.base_actor, IRUNetwork)
                self.assertTrue(agent.base_actor_trainable)
                if adapter_mode == 'none':
                    self.assertIsNone(agent.actor_hypernet)
                else:
                    self.assertIsNotNone(agent.actor_hypernet)
                    film_params = agent.actor_hypernet(agent._agent_meta(1))
                    torch.testing.assert_close(film_params, torch.zeros_like(film_params))

                agent.optimizer.zero_grad()
                valid_logits = logits.masked_select(agent.action_mask.unsqueeze(0))
                (valid_logits.mean() + values.mean()).backward()
                actor_grads = [param.grad for param in agent.base_actor.parameters()]
                self.assertTrue(
                    any(
                        grad is not None
                        and torch.isfinite(grad).all()
                        and float(grad.abs().sum()) > 0.0
                        for grad in actor_grads
                    )
                )
                if adapter_mode == 'film':
                    film_grads = [param.grad for param in agent.actor_hypernet.parameters()]
                    self.assertTrue(
                        any(
                            grad is not None
                            and torch.isfinite(grad).all()
                            and float(grad.abs().sum()) > 0.0
                            for grad in film_grads
                        )
                    )
                signatures[adapter_mode] = agent._architecture_signature()
        finally:
            Registry.mapping['model_mapping']['setting'] = SimpleNamespace(param=original_cfg)

        for key in (
            'graph_critic_enabled',
            'graph_critic_hidden_dim',
            'graph_critic_layers',
            'graph_critic_heads',
            'centralized_critic_mode',
            'value_hidden',
            'value_hypernet_type',
        ):
            self.assertEqual(signatures['none'][key], signatures['film'][key])

    def test_shared_and_film_iru_checkpoints_round_trip(self):
        original_cfg = Registry.mapping['model_mapping']['setting'].param
        try:
            with tempfile.TemporaryDirectory() as output_root:
                for adapter_idx, adapter_mode in enumerate(('none', 'film')):
                    cfg = dict(original_cfg)
                    cfg.update(
                        {
                            'hyper_actor_arch': 'iru',
                            'hyper_adapter_mode': adapter_mode,
                            'hyper_residual': False,
                            'iru_actor_hidden_dim': 16,
                            'iru_actor_steps': 1,
                            'iru_num_blocks': 1,
                        }
                    )
                    Registry.mapping['model_mapping']['setting'] = SimpleNamespace(param=cfg)
                    output_dir = os.path.join(output_root, adapter_mode)
                    os.makedirs(os.path.join(output_dir, 'model'))
                    Registry.mapping['logger_mapping']['path'] = SimpleNamespace(path=output_dir)

                    agent = HyperLightGraphMAPPOAgent(self.world, 0)
                    state = self._state(agent)
                    logits, values = agent._policy_value(state)
                    valid_logits = logits.masked_select(agent.action_mask.unsqueeze(0))
                    agent.optimizer.zero_grad()
                    (valid_logits.mean() + values.mean()).backward()
                    agent.optimizer.step()
                    with torch.no_grad():
                        expected_logits, expected_values = agent._policy_value(state)
                    agent.save_model(20 + adapter_idx)

                    restored = HyperLightGraphMAPPOAgent(self.world, 0)
                    restored.load_model(20 + adapter_idx)
                    with torch.no_grad():
                        actual_logits, actual_values = restored._policy_value(state)
                    torch.testing.assert_close(expected_logits, actual_logits)
                    torch.testing.assert_close(expected_values, actual_values)
                    self.assertTrue(restored.optimizer.state_dict()['state'])
        finally:
            Registry.mapping['model_mapping']['setting'] = SimpleNamespace(param=original_cfg)

    def test_actor_and_pooled_critic_film_identity_gradients_and_checkpoint(self):
        original_cfg = Registry.mapping['model_mapping']['setting'].param
        try:
            cfg = dict(original_cfg)
            cfg.update(
                {
                    'movement_encoder': False,
                    'graph_critic': False,
                    'centralized_critic': True,
                    'centralized_critic_mode': 'pooled',
                    'hyper_actor_arch': 'iru',
                    'hyper_adapter_mode': 'film',
                    'hyper_critic_adapter_mode': 'film',
                    'hyper_residual': False,
                    'iru_actor_hidden_dim': 16,
                    'iru_actor_steps': 1,
                    'iru_num_blocks': 1,
                    'value_hidden': [12, 8],
                }
            )
            Registry.mapping['model_mapping']['setting'] = SimpleNamespace(param=cfg)
            agent = HyperLightPPOAgent(self.world, 0)
            state = self._state(agent)
            policy_state = agent._encode_policy_state(state)
            value_input = agent._value_input(policy_state)
            meta = agent._agent_meta(1)

            actor_film = agent.actor_hypernet(meta)
            value_film = agent.value_hypernet(meta)
            torch.testing.assert_close(actor_film, torch.zeros_like(actor_film))
            torch.testing.assert_close(value_film, torch.zeros_like(value_film))
            self.assertEqual(2 * (12 + 8), agent.value_film_param_dim)

            logits, values, diagnostics = agent._policy_value(
                state,
                return_residual_diagnostics=True,
            )
            with torch.no_grad():
                shared_values = agent.base_value(value_input).squeeze(-1)
            torch.testing.assert_close(values, shared_values)
            self.assertEqual(1.0, diagnostics['hyper_adapter_is_film'])
            self.assertEqual(1.0, diagnostics['critic_adapter_is_film'])

            agent.optimizer.zero_grad()
            valid_logits = logits.masked_select(agent.action_mask.unsqueeze(0))
            (valid_logits.mean() + values.mean()).backward()
            for module in (
                agent.base_actor,
                agent.actor_hypernet,
                agent.base_value,
                agent.value_hypernet,
            ):
                gradients = [param.grad for param in module.parameters() if param.requires_grad]
                self.assertTrue(
                    any(
                        grad is not None
                        and torch.isfinite(grad).all()
                        and float(grad.abs().sum()) > 0.0
                        for grad in gradients
                    )
                )

            with tempfile.TemporaryDirectory() as output_dir:
                Registry.mapping['logger_mapping']['path'] = SimpleNamespace(path=output_dir)
                os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
                agent.optimizer.step()
                with torch.no_grad():
                    expected_logits, expected_values = agent._policy_value(state)
                agent.save_model(31)
                restored = HyperLightPPOAgent(self.world, 0)
                restored.load_model(31)
                with torch.no_grad():
                    actual_logits, actual_values = restored._policy_value(state)
                torch.testing.assert_close(expected_logits, actual_logits)
                torch.testing.assert_close(expected_values, actual_values)
        finally:
            Registry.mapping['model_mapping']['setting'] = SimpleNamespace(param=original_cfg)

    def test_checkpoint_round_trip_restores_new_architecture(self):
        with tempfile.TemporaryDirectory() as output_dir:
            Registry.mapping['logger_mapping']['path'] = SimpleNamespace(path=output_dir)
            os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
            agent = HyperLightGraphMAPPOAgent(self.world, 0)
            state = self._state(agent)
            logits, values = agent._policy_value(state)
            valid_logits = logits.masked_select(agent.action_mask.unsqueeze(0))
            agent.optimizer.zero_grad()
            (valid_logits.mean() + values.mean()).backward()
            agent.optimizer.step()
            with torch.no_grad():
                expected_logits, expected_values = agent._policy_value(state)
            agent.save_model(7)

            restored = HyperLightGraphMAPPOAgent(self.world, 0)
            restored.load_model(7)
            with torch.no_grad():
                actual_logits, actual_values = restored._policy_value(state)
            torch.testing.assert_close(expected_logits, actual_logits)
            torch.testing.assert_close(expected_values, actual_values)
            self.assertTrue(restored.optimizer.state_dict()['state'])

    def test_checkpoint_signature_rejects_shape_compatible_config_drift(self):
        agent = HyperLightGraphMAPPOAgent(self.world, 0)
        checkpoint = {'architecture': agent._architecture_signature()}
        agent.graph_critic_heads = 8
        with self.assertRaisesRegex(RuntimeError, 'graph_critic_heads'):
            agent._validate_checkpoint_architecture(checkpoint)

    def test_synthetic_ppo_update_keeps_raw_states_and_updates_new_modules(self):
        np.random.seed(22)
        torch.manual_seed(22)
        agent = HyperLightGraphMAPPOAgent(self.world, 0)
        agent.ppo_rollout_steps = 2
        agent.ppo_epochs = 1
        agent.ppo_minibatch_size = 6
        obs = agent.get_ob()
        phase = agent.get_phase()

        for step in range(2):
            actions = agent.get_action(obs, phase)
            probabilities = agent.get_action_prob(obs, phase)
            next_obs = obs + np.float32(0.01 * (step + 1))
            next_phase = (phase + 1) % agent.phase_lengths
            agent.remember(
                obs,
                phase,
                actions,
                probabilities,
                np.asarray([-0.1, -0.2, -0.3], dtype=np.float32),
                next_obs,
                next_phase,
                step == 1,
                'step',
            )
            obs, phase = next_obs, next_phase

        buffered_state, buffered_next_state = agent.rollout_buffer[0][:2]
        self.assertEqual((3, agent.raw_state_dim), tuple(buffered_state.shape))
        self.assertFalse(np.array_equal(buffered_state, buffered_next_state))
        before_movement = [param.detach().clone() for param in agent.movement_encoder.parameters()]
        before_graph = [param.detach().clone() for param in agent.graph_critic.parameters()]

        loss = agent.train()
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(before_movement, agent.movement_encoder.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(before_graph, agent.graph_critic.parameters())
            )
        )
        self.assertEqual(1.0, agent.get_residual_diagnostics()['hyper_adapter_is_film'])


if __name__ == '__main__':
    unittest.main()
