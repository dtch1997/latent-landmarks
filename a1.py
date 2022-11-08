from latent_landmarks.unitree_a1 import A1Env

env = A1Env(render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation)
    if terminated or truncated:
        observation, info = env.reset()