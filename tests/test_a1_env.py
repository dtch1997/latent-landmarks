import latent_landmarks.envs
import gymnasium as gym

def test_reset():
    env = gym.make("A1-v1")
    env.reset(seed=42)

def test_obs_space():
    env = gym.make("A1-v1")
    observation, info = env.reset(seed=42)
    assert observation.shape == (35,)