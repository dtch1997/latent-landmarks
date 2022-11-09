import gymnasium as gym
import latent_landmarks.envs # Register envs

def test_reset():
    env = gym.make("GoalA1-v1")
    env.reset(seed=42)

def test_obs_space():
    env = gym.make("GoalA1-v1")
    observation, info = env.reset(seed=42)
    assert observation['observation'].shape == (35,)
    assert observation['desired_goal'].shape == (12,)
    assert observation['achieved_goal'].shape == (12,)