import numpy as np
import torch
import datetime
import os
import os.path as osp
import sys

from rl import logger

from rl.utils import mpi_utils
from rl.utils.run_utils import Timer, log_config, merge_configs
from rl.replay.core import sample_her_transitions


class BaseAlgo:
    def __init__(
            self,
            env, env_params, args,
            agent, replay, monitor, learner,
            reward_func,
            name='algo',
    ):
        self.env = env
        self.env_params = env_params
        self.args = args
        
        self.agent = agent
        self.replay = replay
        self.monitor = monitor
        self.learner = learner
        
        self.reward_func = reward_func
        
        self.timer = Timer()
        self.start_time = self.timer.current_time
        self.total_timesteps = 0
        
        self.env_steps = 0
        self.opt_steps = 0
        
        self.num_envs = 1
        if hasattr(self.env, 'num_envs'):
            self.num_envs = getattr(self.env, 'num_envs')
        
        self.n_mpi = mpi_utils.get_size()
        self._save_file = str(name) + '.pt'
        
        if len(args.resume_ckpt) > 0:
            resume_path = osp.join(
                osp.join(self.args.save_dir, self.args.env_name),
                osp.join(args.resume_ckpt, 'state'))
            self.load_all(resume_path)
        
        self.log_path = osp.join(osp.join(self.args.save_dir, self.args.env_name), args.ckpt_name)
        self.model_path = osp.join(self.log_path, 'state')
        if mpi_utils.is_root() and not args.play:
            os.makedirs(self.model_path, exist_ok=True)
            logger.configure(dir=self.log_path, format_strs=["csv", "stdout"])
            config_list = [env_params.copy(), args.__dict__.copy(), {'NUM_MPI': mpi_utils.get_size()}]
            log_config(config=merge_configs(config_list), output_dir=self.log_path)
    
    def run_eval(self, use_test_env=False):
        env = self.env
        if use_test_env and hasattr(self, 'test_env'):
            env = self.test_env
        total_success_count = 0
        total_trial_count = 0
        for n_test in range(self.args.n_test_rollouts):
            seed = self.env_params['reset_seed']
            observation, info = env.reset(seed=seed)
            ob = observation['observation']
            bg = observation['desired_goal']
            ag = observation['achieved_goal']
            ag_origin = ag.copy()
            for timestep in range(env._max_episode_steps):
                act = self.agent.get_actions(ob, bg)
                observation, reward, terminated, truncated, info = self.env.step(act)
                ob = observation['observation']
                bg = observation['desired_goal']
                ag = observation['achieved_goal']
                ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
                self.monitor.store(Inner_Test_AgChangeRatio=np.mean(ag_changed))
            ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
            self.monitor.store(TestAgChangeRatio=np.mean(ag_changed))
            if self.num_envs > 1:
                for per_env_info in info:
                    total_trial_count += 1
                    if per_env_info['is_success'] == 1.0:
                        total_success_count += 1
            else:
                total_trial_count += 1
                if info['is_success'] == 1.0:
                    total_success_count += 1
        success_rate = total_success_count / total_trial_count
        if mpi_utils.use_mpi():
            success_rate = mpi_utils.global_mean(np.array([success_rate]))[0]
        return success_rate
    
    def log_everything(self):
        for log_name in self.monitor.epoch_dict:
            log_item = self.monitor.log(log_name)
            if mpi_utils.use_mpi():
                log_item_k = log_item.keys()
                log_item_v = np.array(list(log_item.values()))
                log_item_v = mpi_utils.global_mean(log_item_v)
                log_item = dict(zip(log_item_k, log_item_v))
            logger.record_tabular(log_name, log_item['mean'])
        logger.record_tabular('TotalTimeSteps', self.total_timesteps)
        logger.record_tabular('Time', self.timer.current_time - self.start_time)
        if mpi_utils.is_root():
            logger.dump_tabular()
    
    def state_dict(self):
        raise NotImplementedError
    
    def load_state_dict(self, state_dict):
        raise NotImplementedError
    
    def save(self, path):
        if mpi_utils.is_root():
            state_dict = self.state_dict()
            save_path = osp.join(path, self._save_file)
            torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        try:
            state_dict = torch.load(load_path)
        except RuntimeError:
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
    
    def save_all(self, path):
        self.save(path)
        self.agent.save(path)
        self.replay.save(path)
        self.learner.save(path)
    
    def load_all(self, path):
        self.load(path)
        self.agent.load(path)
        self.replay.load(path)
        self.learner.load(path)