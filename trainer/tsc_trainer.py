import logging
import os
import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer


@Registry.register_trainer("tsc")
class TSCTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''
    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="tsc"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        logger_params = Registry.mapping['logger_mapping']['setting'].param
        self.save_rate = int(logger_params['save_rate'])
        self.train_log_interval = max(1, int(logger_params.get('train_log_interval', 1)))
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        self.test_interval = max(
            1,
            int(Registry.mapping['trainer_mapping']['setting'].param.get('test_interval', 1))
        )
        self.early_stop_patience = int(
            Registry.mapping['trainer_mapping']['setting'].param.get('early_stop_patience', 0)
        )
        self.load_best_for_test = bool(
            Registry.mapping['trainer_mapping']['setting'].param.get('load_best_for_test', True)
        )
        self.resume_episode = int(
            Registry.mapping['trainer_mapping']['setting'].param.get('resume_episode', 0) or 0
        )
        if self.resume_episode < 0:
            raise ValueError('trainer.resume_episode must be non-negative')
        if self.resume_episode >= self.episodes:
            raise ValueError(
                'trainer.resume_episode must be smaller than trainer.episodes '
                f'(got {self.resume_episode} >= {self.episodes})'
            )
        self.best_test_travel_time = float('inf')
        self.best_test_episode = -1
        self.no_improve_rounds = 0
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later
        
        # TODO: support other dataset in the future
        self.dataset = Registry.mapping['dataset_mapping'][Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip('_BRF.log') + '_DTL.log'
                                     )
        self.cos_log_file = self.log_file.replace('_DTL.log', '_COS.log')
        self.residual_log_file = self.log_file.replace('_DTL.log', '_RES.log')
        self.performance_log_file = self.log_file.replace('_DTL.log', '_PERF.log')

    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''
        self.agents = []
        agent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, 0)
        print(agent)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, i))

        # for magd agents should share information 
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env = TSCEnv(self.world, self.agents, self.metric)

    def train(self):
        '''
        train
        Train the agent(s).

        :param: None
        :return: None
        '''
        if self.resume_episode > 0:
            [ag.load_model(self.resume_episode) for ag in self.agents]
            self.logger.info(
                'Resumed training from episode %d; continuing through episode %d',
                self.resume_episode,
                self.episodes - 1,
            )

        decisions_per_episode = max(1, self.steps // self.action_interval)
        total_decision_num = self.resume_episode * decisions_per_episode
        flush = 0
        for e in range(self.resume_episode, self.episodes):
            # TODO: check this reset agent
            self.metric.clear()
            last_obs = self.env.reset()  # agent * [sub_agent, feature]

            for a in self.agents:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and self.save_rate > 0 and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]

                    if total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents):
                            actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))                            
                        actions = np.stack(actions)  # [agent, intersections]
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents])

                    actions_prob = []
                    for idx, ag in enumerate(self.agents):
                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    self.metric.update(rewards)

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])
                    for idx, ag in enumerate(self.agents):
                        ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                            obs[idx], cur_phase[idx], dones[idx], f'{e}_{i//self.action_interval}_{ag.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
                    total_decision_num += 1
                    last_obs = obs
                if total_decision_num > self.learning_start and\
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:

                    cur_loss_q = np.stack([ag.train() for ag in self.agents])  # TODO: training

                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents]

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            
            should_log_train = (e % self.train_log_interval == 0) or (e == self.episodes - 1)
            if should_log_train:
                travel_time = self.metric.real_average_travel_time()
                mean_reward = self.metric.rewards()
                mean_queue = self.metric.queue()
                mean_delay = self.metric.delay()
                throughput = self.metric.throughput()
                self.writeLog("TRAIN", e, travel_time, mean_loss, mean_reward, mean_queue, mean_delay, throughput)
                cos_diagnostics = self._collect_cos_diagnostics(source='episode')
                if cos_diagnostics:
                    self.writeCosLog("TRAIN", e, cos_diagnostics)
                    self.logger.info(
                        "cos_diagnostics: {}".format(self._format_cos_diagnostics(cos_diagnostics))
                    )
                residual_diagnostics = self._collect_residual_diagnostics(source='episode')
                if residual_diagnostics:
                    self.writeResidualLog("TRAIN", e, residual_diagnostics)
                    self.logger.info(
                        "residual_diagnostics: {}".format(
                            self._format_residual_diagnostics(residual_diagnostics)
                        )
                    )
                residual_update_diagnostics = self._collect_residual_diagnostics(source='update')
                if residual_update_diagnostics:
                    self.writeResidualLog("TRAIN_UPDATE", e, residual_update_diagnostics)
                    self.logger.info(
                        "residual_update_diagnostics: {}".format(
                            self._format_residual_diagnostics(residual_update_diagnostics)
                        )
                    )
                performance_diagnostics = self._collect_performance_diagnostics()
                if performance_diagnostics:
                    self.writePerformanceLog("TRAIN", e, performance_diagnostics)
                    self.logger.info(
                        "performance_diagnostics: {}".format(
                            self._format_performance_diagnostics(performance_diagnostics)
                        )
                    )
                self.logger.info(
                    "step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(
                        i, self.steps, mean_loss, mean_reward, mean_queue, mean_delay, int(throughput)
                    )
                )
                self.logger.info("episode:{}/{}, real avg travel time:{}".format(e, self.episodes, travel_time))
            if self.save_rate > 0 and e % self.save_rate == 0:
                [ag.save_model(e=e) for ag in self.agents]
            if should_log_train and self.logger.isEnabledFor(logging.DEBUG):
                lane_rewards = self.metric.lane_rewards()
                lane_queues = self.metric.lane_queue()
                for j in range(len(self.world.intersections)):
                    self.logger.debug(
                        "intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(
                            j, lane_rewards[j], lane_queues[j]
                        )
                    )
            if self.test_when_train and e % self.test_interval == 0:
                test_travel_time = self.train_test(e, mean_loss)
                if test_travel_time + 1e-6 < self.best_test_travel_time:
                    self.best_test_travel_time = test_travel_time
                    self.best_test_episode = e
                    self.no_improve_rounds = 0
                    [ag.save_model(e='best') for ag in self.agents]
                    self.logger.info(
                        "New best TEST travel time %.4f at episode %d, saved as best checkpoint",
                        self.best_test_travel_time,
                        e,
                    )
                else:
                    self.no_improve_rounds += 1
                    if self.early_stop_patience > 0 and self.no_improve_rounds >= self.early_stop_patience:
                        self.logger.info(
                            "Early stop triggered at episode %d (best episode %d, best TEST travel time %.4f)",
                            e,
                            self.best_test_episode,
                            self.best_test_travel_time,
                        )
                        break
        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
        [ag.save_model(e=self.episodes) for ag in self.agents]

    def train_test(self, e, train_loss=np.nan):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        obs = self.env.reset()
        self.metric.clear()
        for a in self.agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break
        travel_time = self.metric.real_average_travel_time()
        mean_reward = self.metric.rewards()
        mean_queue = self.metric.queue()
        mean_delay = self.metric.delay()
        throughput = self.metric.throughput()
        self.logger.info(
            "Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(
                e, self.episodes, travel_time, mean_reward, mean_queue, mean_delay, int(throughput)
            )
        )
        self.writeLog(
            "TEST", e, travel_time, train_loss, mean_reward, mean_queue, mean_delay, throughput
        )
        cos_diagnostics = self._collect_cos_diagnostics(source='episode')
        if cos_diagnostics:
            self.writeCosLog("TEST", e, cos_diagnostics)
            self.logger.info(
                "test_cos_diagnostics: {}".format(self._format_cos_diagnostics(cos_diagnostics))
            )
        residual_diagnostics = self._collect_residual_diagnostics(source='episode')
        if residual_diagnostics:
            self.writeResidualLog("TEST", e, residual_diagnostics)
            self.logger.info(
                "test_residual_diagnostics: {}".format(
                    self._format_residual_diagnostics(residual_diagnostics)
                )
            )
        performance_diagnostics = self._collect_performance_diagnostics()
        if performance_diagnostics:
            self.writePerformanceLog("TEST", e, performance_diagnostics)
            self.logger.info(
                "test_performance_diagnostics: {}".format(
                    self._format_performance_diagnostics(performance_diagnostics)
                )
            )
        return travel_time

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()
        if not drop_load:
            loaded_best = False
            if self.load_best_for_test:
                try:
                    [ag.load_model('best') for ag in self.agents]
                    loaded_best = True
                    self.logger.info("Loaded best checkpoint for final evaluation")
                except Exception:
                    loaded_best = False
            if not loaded_best:
                [ag.load_model(self.episodes) for ag in self.agents]
        attention_mat_list = []
        obs = self.env.reset()
        for a in self.agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break
        travel_time = self.metric.real_average_travel_time()
        mean_reward = self.metric.rewards()
        mean_queue = self.metric.queue()
        mean_delay = self.metric.delay()
        throughput = self.metric.throughput()
        self.logger.info(
            "Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d"
            % (travel_time, mean_reward, mean_queue, mean_delay, throughput)
        )
        return self.metric

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

    def _collect_cos_diagnostics(self, source='update'):
        diagnostics = []
        for agent in self.agents:
            method_name = 'get_cos_episode_diagnostics' if source == 'episode' else 'get_cos_diagnostics'
            get_diagnostics = getattr(agent, method_name, None)
            if not callable(get_diagnostics):
                continue
            agent_diagnostics = get_diagnostics()
            if agent_diagnostics:
                diagnostics.append(agent_diagnostics)
        return self._mean_cos_diagnostics(diagnostics)

    def _collect_residual_diagnostics(self, source='update'):
        diagnostics = []
        for agent in self.agents:
            method_name = (
                'get_residual_episode_diagnostics'
                if source == 'episode'
                else 'get_residual_diagnostics'
            )
            get_diagnostics = getattr(agent, method_name, None)
            if not callable(get_diagnostics):
                continue
            agent_diagnostics = get_diagnostics()
            if agent_diagnostics:
                diagnostics.append(agent_diagnostics)
        return self._mean_cos_diagnostics(diagnostics)

    def _collect_performance_diagnostics(self):
        diagnostics = []
        for agent in self.agents:
            get_diagnostics = getattr(agent, 'get_performance_diagnostics', None)
            if not callable(get_diagnostics):
                continue
            agent_diagnostics = get_diagnostics()
            if agent_diagnostics:
                diagnostics.append(agent_diagnostics)
        return self._mean_cos_diagnostics(diagnostics)

    @staticmethod
    def _cos_diagnostic_keys():
        return [
            'cos_entropy',
            'cos_self_selection_rate',
            'cos_avg_selected_hop',
            'cos_avg_selected_distance',
            'cos_diag_mass',
            'cos_symmetry_loss',
        ]

    def _format_cos_diagnostics(self, diagnostics):
        return ', '.join(
            '{}:{:.6f}'.format(key, diagnostics[key])
            for key in self._cos_diagnostic_keys()
            if key in diagnostics
        )

    @staticmethod
    def _residual_diagnostic_keys():
        return [
            'hyper_adapter_is_film',
            'film_scale',
            'film_param_dim',
            'film_gamma_abs_mean',
            'film_beta_abs_mean',
            'hyper_residual_actor_scale',
            'hyper_residual_value_scale',
            'hyper_residual_is_lora',
            'hyper_residual_is_head',
            'hyper_head_actor_param_dim',
            'hyper_head_value_param_dim',
            'hyper_lora_actor_rank',
            'hyper_lora_value_rank',
            'actor_base_norm',
            'actor_delta_norm',
            'actor_delta_base_ratio',
            'actor_delta_max_abs',
            'actor_theta_norm',
            'actor_head_base_norm',
            'actor_head_delta_norm',
            'actor_head_delta_base_ratio',
            'actor_head_delta_max_abs',
            'actor_head_theta_norm',
            'value_base_norm',
            'value_delta_norm',
            'value_delta_base_ratio',
            'value_delta_max_abs',
            'value_theta_norm',
            'value_head_base_norm',
            'value_head_delta_norm',
            'value_head_delta_base_ratio',
            'value_head_delta_max_abs',
            'value_head_theta_norm',
            'policy_logit_std',
            'policy_logit_abs_mean',
            'value_std',
            'value_abs_mean',
            'meta_norm',
            'meta_std',
        ]

    def _format_residual_diagnostics(self, diagnostics):
        return ', '.join(
            '{}:{:.6f}'.format(key, diagnostics[key])
            for key in self._residual_diagnostic_keys()
            if key in diagnostics
        )

    @staticmethod
    def _performance_diagnostic_keys():
        return [
            'parameter_count',
            'actor_parameter_count',
            'value_parameter_count',
            'embedding_parameter_count',
            'decision_count',
            'decision_latency_ms_mean',
            'decision_time_ms_total',
            'update_count',
            'update_time_ms_mean',
            'update_time_ms_total',
            'gpu_peak_memory_mb',
            'gpu_peak_reserved_mb',
        ]

    def _format_performance_diagnostics(self, diagnostics):
        return ', '.join(
            '{}:{:.6f}'.format(key, diagnostics[key])
            for key in self._performance_diagnostic_keys()
            if key in diagnostics
        )

    def writeCosLog(self, mode, step, diagnostics):
        if not diagnostics:
            return

        keys = self._cos_diagnostic_keys()
        needs_header = (not os.path.exists(self.cos_log_file)) or os.path.getsize(self.cos_log_file) == 0
        with open(self.cos_log_file, "a") as log_handle:
            if needs_header:
                log_handle.write("model\tmode\tstep\t" + "\t".join(keys) + "\n")
            res = (
                Registry.mapping['model_mapping']['setting'].param['name']
                + '\t' + mode
                + '\t' + str(step)
                + '\t' + '\t'.join(
                    "{:.6f}".format(diagnostics[key]) if key in diagnostics else ""
                    for key in keys
                )
            )
            log_handle.write(res + "\n")

    def writeResidualLog(self, mode, step, diagnostics):
        if not diagnostics:
            return

        keys = self._residual_diagnostic_keys()
        needs_header = (
            (not os.path.exists(self.residual_log_file))
            or os.path.getsize(self.residual_log_file) == 0
        )
        with open(self.residual_log_file, "a") as log_handle:
            if needs_header:
                log_handle.write("model\tmode\tstep\t" + "\t".join(keys) + "\n")
            res = (
                Registry.mapping['model_mapping']['setting'].param['name']
                + '\t' + mode
                + '\t' + str(step)
                + '\t' + '\t'.join(
                    "{:.6f}".format(diagnostics[key]) if key in diagnostics else ""
                    for key in keys
                )
            )
            log_handle.write(res + "\n")

    def writePerformanceLog(self, mode, step, diagnostics):
        if not diagnostics:
            return

        keys = self._performance_diagnostic_keys()
        needs_header = (
            (not os.path.exists(self.performance_log_file))
            or os.path.getsize(self.performance_log_file) == 0
        )
        with open(self.performance_log_file, "a") as log_handle:
            if needs_header:
                log_handle.write("model\tmode\tstep\t" + "\t".join(keys) + "\n")
            res = (
                Registry.mapping['model_mapping']['setting'].param['name']
                + '\t' + mode
                + '\t' + str(step)
                + '\t' + '\t'.join(
                    "{:.6f}".format(diagnostics[key]) if key in diagnostics else ""
                    for key in keys
                )
            )
            log_handle.write(res + "\n")

    def writeLog(self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        '''
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''
        res = (
            Registry.mapping['model_mapping']['setting'].param['name']
            + '\t' + mode
            + '\t' + str(step)
            + '\t' + f"{travel_time:.4f}"
            + '\t' + f"{loss:.6f}"
            + "\t" + f"{cur_rwd:.4f}"
            + "\t" + f"{cur_queue:.4f}"
            + "\t" + f"{cur_delay:.4f}"
            + "\t" + f"{int(cur_throughput)}"
        )
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()
