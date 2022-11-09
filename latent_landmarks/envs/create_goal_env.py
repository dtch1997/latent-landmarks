from latent_landmarks.envs.unitree_a1 import A1Env
from collections import OrderedDict
import gymnasium as gym
import numpy as np
import copy

from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec

class GoalWrapper(gym.Wrapper):
    def __init__(self, env, random_start, distance_threshold = 0.1):
        super(GoalWrapper, self).__init__(env)
        ob_space = env.observation_space
        goal_space = env.action_space

        self.goal_space = env.action_space
        self.goal_dim = goal_space.shape[0]
        self.distance_threshold = distance_threshold 
        
        self.observation_space = spaces.Dict(OrderedDict({
            'observation': ob_space,
            'desired_goal': self.goal_space,
            'achieved_goal': self.goal_space,
        }))
        self.goal = None
        self.random_start = random_start
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        achieved_goal = self.unwrapped.get_joint_pos()
        out = {'observation': observation,
               'desired_goal': self.goal,
               'achieved_goal': achieved_goal }
        goal_distance = np.linalg.norm(achieved_goal - self.goal, axis=-1)
        info['is_success'] = (goal_distance < self.distance_threshold)
        reward = self.compute_rew(achieved_goal, self.goal)
        return out, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.goal = self.goal_space.sample()
        

        achieved_goal = self.unwrapped.get_joint_pos()
        goal_distance = np.linalg.norm(achieved_goal - self.goal, axis=-1)
        while goal_distance < self.distance_threshold:
            self.goal = self.goal_space.sample()
        info['is_success'] = False

        # randomly initialize a joint pose
        if self.random_start:
            raise NotImplementedError
        
        out = dict(observation=observation, desired_goal=self.goal, achieved_goal=self.unwrapped.get_joint_pos())
        return out, info
    
    def compute_rew(self, state, goal):
        assert state.shape == goal.shape
        dist = np.linalg.norm(state - goal, axis=-1)
        return -(dist > self.distance_threshold).astype(np.float32)
    
    def render(self, *args, **kwargs):
        self.env.wrapped_env.render_callback(self.goal)
        return self.env.render(*args, **kwargs)


def create_goal_env(random_start=False):
    return GoalWrapper(A1Env(), random_start)