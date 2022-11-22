import gym
import argparse
import random

def main(env_name):
    env = gym.make(env_name, render_mode='human')
    obs, _ = env.reset()
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    while True:
        env.render()
        action = random.choice(range(action_dim))

        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', default='MountainCar-v0')
    args = parser.parse_args()
    env_name = args.env

    main(env_name)