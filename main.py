import gym
import argparse
import random
from Buffer import replay_buffer
from Agent import DQN

def main(env_name):
    env = gym.make(env_name)
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    episodes = 10000
    buffer = replay_buffer.ReplayBuffer()
    agent = DQN.DQNAgent(action_dim, obs_dim)

    counter = 0

    end_count = 300

    while True:
        obs, _ = env.reset()
        total_reward = 0
        while True:
            action = agent.step(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            

            buffer.insert([obs, reward, action, next_obs, terminated or truncated])
            if terminated or truncated:
                break

            obs = next_obs

            if len(buffer) >= agent.batch_size:  
                agent.learn(buffer)


        agent.eps_scheduler()
        agent.soft_update()

        counter += 1
        print(f'Episode: {counter}, reward: {total_reward}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', default='MountainCar-v0')
    args = parser.parse_args()
    env_name = args.env

    main(env_name)