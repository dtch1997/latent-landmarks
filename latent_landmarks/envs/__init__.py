import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='GoalA1-v1',
    entry_point='latent_landmarks.envs.goal_unitree_a1:GoalA1Env',
    max_episode_steps=100,
)