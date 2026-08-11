import copy
import os
import tempfile
import unittest
from collections import deque
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn

from agent.maddpg_v2 import (
    ActorNetwork,
    CriticNetwork,
    MADDPGAgent,
    MADDPG_SUBAgent,
    _align_action_probabilities,
)
from agent.dqn import DQNAgent
from agent.frap import FRAP_DQNAgent
from agent.magd import MAGDAgent
from agent.mplight import MPLightAgent
from agent.ppo_pfrl import IPPO_pfrl
from agent.presslight import PressLightAgent
from common.registry import Registry
from world.world_cityflow import World


class _FakeEngine:
    def __init__(self, distances):
        self.distances = distances
        self.distance_calls = 0

    def get_vehicle_distance(self):
        self.distance_calls += 1
        return dict(self.distances)


class RealDelayTests(unittest.TestCase):
    def test_real_delay_is_idempotent_and_does_not_advance_trajectories(self):
        world = World.__new__(World)
        world.eng = _FakeEngine({'vehicle_1': 50.0, 'vehicle_2': 30.0})
        world.all_lanes_speed = {'lane_a': 10.0, 'lane_b': 10.0}
        world.lane_length = {'lane_a': 100.0, 'lane_b': 80.0}
        world.vehicle_trajectory = {
            'vehicle_1': [['lane_a', 0, 20]],
            'vehicle_2': [['lane_a', 0, 8], ['lane_b', 8, 10]],
        }
        world.real_delay = {'stale_value': 999.0}
        original_trajectories = copy.deepcopy(world.vehicle_trajectory)

        first = world.get_real_delay()
        second = world.get_real_delay()

        # vehicle_1: 20 - 50/10 = 15; vehicle_2: 0 + (10 - 30/10) = 7
        self.assertAlmostEqual(first, 11.0)
        self.assertEqual(second, first)
        self.assertEqual(world.vehicle_trajectory, original_trajectories)
        self.assertEqual(world.real_delay, {'stale_value': 999.0})

    def test_real_delay_is_zero_for_an_empty_episode(self):
        world = World.__new__(World)
        world.eng = _FakeEngine({})
        world.vehicle_trajectory = {}

        self.assertEqual(world.get_real_delay(), 0.0)
        self.assertEqual(world.eng.distance_calls, 0)


class _ActionStub:
    def __init__(self, probabilities):
        self.action_space = SimpleNamespace(n=len(probabilities))
        self.probabilities = np.asarray(probabilities, dtype=np.float32)
        self.choose_calls = 0

    def choose_action(self, observation, test=False):
        self.choose_calls += 1
        return self.probabilities.copy()


class MADDPGActionTests(unittest.TestCase):
    def test_get_action_prob_returns_the_exact_cached_gumbel_sample(self):
        sub_agent = _ActionStub([0.1, 0.8, 0.1])
        agent = MADDPGAgent.__new__(MADDPGAgent)
        agent.agents = [sub_agent]
        agent.sub_agents = 1
        agent.prob = []
        agent.replay_buffer = deque()

        actions = agent.get_action([np.array([1.0])], phase=None)
        probabilities = agent.get_action_prob(None, None)
        probabilities_again = agent.get_action_prob(None, None)

        self.assertEqual(actions, [1])
        self.assertEqual(sub_agent.choose_calls, 1)
        np.testing.assert_array_equal(probabilities, probabilities_again)
        self.assertEqual(int(np.argmax(probabilities[0])), actions[0])

        agent.remember(None, None, actions, probabilities, None, None, None, False, 'step')
        replay_probabilities = agent.replay_buffer[-1][1][2]
        np.testing.assert_array_equal(replay_probabilities[0], probabilities[0])

    def test_replay_falls_back_to_executed_action_when_soft_action_disagrees(self):
        aligned = _align_action_probabilities(
            actions=[2],
            action_probabilities=[[0.9, 0.05, 0.05]],
            action_sizes=[3],
        )

        np.testing.assert_array_equal(aligned[0], [0.0, 0.0, 1.0])

    def test_get_action_prob_remains_callable_without_a_prior_action(self):
        sub_agent = _ActionStub([0.2, 0.3, 0.5])
        agent = MADDPGAgent.__new__(MADDPGAgent)
        agent.agents = [sub_agent]
        agent.prob = []

        probabilities = agent.get_action_prob([np.array([1.0])], phase=None)

        self.assertEqual(sub_agent.choose_calls, 1)
        np.testing.assert_array_equal(probabilities[0], sub_agent.probabilities)

    def test_random_sample_caches_matching_one_hot_actions(self):
        agent = MADDPGAgent.__new__(MADDPGAgent)
        agent.agents = [_ActionStub([0.5, 0.5]), _ActionStub([0.25] * 4)]
        agent.sub_agents = 2
        agent.prob = []

        actions = agent.sample()
        probabilities = agent.get_action_prob(None, None)

        self.assertEqual(len(probabilities), 2)
        for action, probability in zip(actions, probabilities):
            self.assertEqual(int(np.argmax(probability)), int(action))
            self.assertAlmostEqual(float(np.sum(probability)), 1.0)

    def test_bootstrap_networks_do_not_receive_gradients(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            sub_agent = SimpleNamespace()
            sub_agent.actor = ActorNetwork(1e-3, 2, 4, 4, 2, 'actor', checkpoint_dir)
            sub_agent.target_actor = ActorNetwork(
                1e-3, 2, 4, 4, 2, 'target_actor', checkpoint_dir
            )
            sub_agent.critic = CriticNetwork(
                1e-3, 2, 4, 4, 2, 'critic', checkpoint_dir
            )
            sub_agent.target_critic = CriticNetwork(
                1e-3, 2, 4, 4, 2, 'target_critic', checkpoint_dir
            )
            sub_agent.loss = nn.MSELoss()
            sub_agent.gamma = 0.9
            sub_agent.grad_clip = 1.0
            sub_agent.tau = 0.1
            sub_agent.update_network_parameters = MethodType(
                MADDPG_SUBAgent.update_network_parameters, sub_agent
            )

            agent = MADDPGAgent.__new__(MADDPGAgent)
            agent.agents = [sub_agent]
            agent.batch_size = 1
            agent.replay_buffer = deque([
                (
                    'step',
                    (
                        [np.array([0.1, 0.2], dtype=np.float32)],
                        None,
                        [np.array([0.0, 1.0], dtype=np.float32)],
                        [np.array([1.0], dtype=np.float32)],
                        [np.array([0.2, 0.3], dtype=np.float32)],
                        None,
                    ),
                )
            ])

            agent.train()

            self.assertTrue(all(p.grad is None for p in sub_agent.target_actor.parameters()))
            self.assertTrue(all(p.grad is None for p in sub_agent.target_critic.parameters()))


class _FakeInnerAgent:
    def __init__(self):
        self.sync_calls = 0

    def sync_target_network(self):
        self.sync_calls += 1


class MPLightCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_restores_model_optimizer_and_target(self):
        with tempfile.TemporaryDirectory() as output_dir:
            logger_mapping = Registry.mapping['logger_mapping']
            sentinel = object()
            old_path = logger_mapping.get('path', sentinel)
            logger_mapping['path'] = SimpleNamespace(path=output_dir)
            try:
                agent = MPLightAgent.__new__(MPLightAgent)
                agent.rank = 3
                agent.model = nn.Linear(2, 1)
                agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=0.01)

                loss = agent.model(torch.ones(1, 2)).sum()
                loss.backward()
                agent.optimizer.step()
                expected_state = copy.deepcopy(agent.model.state_dict())
                agent.save_model('best')

                def rebuild_model():
                    agent.model = nn.Linear(2, 1)
                    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=0.01)
                    return _FakeInnerAgent()

                agent._build_model = rebuild_model
                agent.load_model('best')

                for name, expected in expected_state.items():
                    torch.testing.assert_close(agent.model.state_dict()[name], expected)
                self.assertTrue(agent.optimizer.state_dict()['state'])
                self.assertEqual(agent.agents_iner.sync_calls, 1)
            finally:
                if old_path is sentinel:
                    logger_mapping.pop('path', None)
                else:
                    logger_mapping['path'] = old_path


class LegacyCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.TemporaryDirectory()
        self.logger_mapping = Registry.mapping['logger_mapping']
        self.sentinel = object()
        self.old_path = self.logger_mapping.get('path', self.sentinel)
        self.logger_mapping['path'] = SimpleNamespace(path=self.output_dir.name)
        os.makedirs(os.path.join(self.output_dir.name, 'model'), exist_ok=True)

    def tearDown(self):
        if self.old_path is self.sentinel:
            self.logger_mapping.pop('path', None)
        else:
            self.logger_mapping['path'] = self.old_path
        self.output_dir.cleanup()

    def test_value_agents_keep_optimizer_bound_to_loaded_model(self):
        for agent_class in (DQNAgent, FRAP_DQNAgent, PressLightAgent):
            with self.subTest(agent=agent_class.__name__):
                agent = agent_class.__new__(agent_class)
                agent.rank = 0
                agent.model = nn.Linear(2, 2)
                agent.target_model = nn.Linear(2, 2)
                agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=0.01)
                model_before_load = agent.model
                optimizer_parameters = [
                    parameter
                    for group in agent.optimizer.param_groups
                    for parameter in group['params']
                ]

                checkpoint_model = nn.Linear(2, 2)
                with torch.no_grad():
                    checkpoint_model.weight.fill_(2.0)
                    checkpoint_model.bias.fill_(3.0)
                torch.save(
                    checkpoint_model.state_dict(),
                    os.path.join(self.output_dir.name, 'model', 'saved_0.pt'),
                )

                agent.load_model('saved')

                self.assertIs(agent.model, model_before_load)
                optimizer_parameters_after_load = [
                    parameter
                    for group in agent.optimizer.param_groups
                    for parameter in group['params']
                ]
                self.assertEqual(
                    [id(parameter) for parameter in optimizer_parameters],
                    [id(parameter) for parameter in optimizer_parameters_after_load],
                )
                self.assertEqual(
                    [id(parameter) for parameter in optimizer_parameters],
                    [id(parameter) for parameter in agent.model.parameters()],
                )
                torch.testing.assert_close(agent.model.weight, checkpoint_model.weight)
                torch.testing.assert_close(agent.target_model.weight, checkpoint_model.weight)

    def test_magd_loads_the_models_used_for_inference_and_syncs_targets(self):
        os.makedirs(os.path.join(self.output_dir.name, 'model_p'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir.name, 'model_q'), exist_ok=True)
        agent = MAGDAgent.__new__(MAGDAgent)
        agent.rank = 1
        agent.p_model = nn.Linear(2, 3)
        agent.target_p_model = nn.Linear(2, 3)
        agent.q_model = nn.Linear(4, 1)
        agent.target_q_model = nn.Linear(4, 1)
        p_model_before_load = agent.p_model
        q_model_before_load = agent.q_model

        saved_p = nn.Linear(2, 3)
        saved_q = nn.Linear(4, 1)
        with torch.no_grad():
            saved_p.weight.fill_(4.0)
            saved_q.weight.fill_(5.0)
        torch.save(
            saved_p.state_dict(),
            os.path.join(self.output_dir.name, 'model_p', 'saved_1.pt'),
        )
        torch.save(
            saved_q.state_dict(),
            os.path.join(self.output_dir.name, 'model_q', 'saved_1.pt'),
        )

        agent.load_model('saved')

        self.assertIs(agent.p_model, p_model_before_load)
        self.assertIs(agent.q_model, q_model_before_load)
        torch.testing.assert_close(agent.p_model.weight, saved_p.weight)
        torch.testing.assert_close(agent.q_model.weight, saved_q.weight)
        torch.testing.assert_close(agent.target_p_model.weight, saved_p.weight)
        torch.testing.assert_close(agent.target_q_model.weight, saved_q.weight)

    def test_pfrl_ppo_loads_from_the_same_logger_path_used_by_save(self):
        agent = IPPO_pfrl.__new__(IPPO_pfrl)
        agent.rank = 2
        saved_model = nn.Linear(2, 1)
        with torch.no_grad():
            saved_model.weight.fill_(6.0)
        torch.save(
            saved_model.state_dict(),
            os.path.join(self.output_dir.name, 'model', 'saved_2.pt'),
        )

        def rebuild_model():
            agent.model = nn.Linear(2, 1)

        agent._build_model = rebuild_model
        agent.load_model('saved')

        torch.testing.assert_close(agent.model.weight, saved_model.weight)


if __name__ == '__main__':
    unittest.main()
