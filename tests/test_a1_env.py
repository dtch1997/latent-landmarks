from latent_landmarks.unitree_a1 import A1Env

def test_reset():
    env = A1Env()
    env.reset(seed=42)

def test_obs_space():
    env = A1Env()
    observation, info = env.reset(seed=42)
    assert observation.shape == (35,)