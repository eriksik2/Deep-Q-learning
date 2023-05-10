import argparse

import gymnasium as gym
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
from gymnasium.wrappers import AtariPreprocessing

import record

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1'], default='Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole,
    'Pong-v5': config.Pong
}

if __name__ == '__main__':
    args = parser.parse_args()


    # Initialize environment and config.
    env = gym.make(args.env if args.env != 'Pong-v5' else 'ALE/Pong-v5')
    if args.env == 'Pong-v5':
        env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    env_config = ENV_CONFIGS[args.env]

    recorder = record.Recorder(args.env, env_config)

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()

        obs = preprocess(obs, env=args.env).unsqueeze(0)
        obs_stack = torch.cat(env_config['obs_stack_size'] * [obs]).unsqueeze(0).to(device)

        while not terminated:
            #GJURT
            # TODO: Get action from DQN.
            action = dqn.act(obs_stack, exploit=False).item()

            # Act in the true environment.
            old_obs_stack = obs_stack
            obs, reward, terminated, truncated, info = env.step(action)

            # Preprocess incoming observation.
            #if not terminated:
            obs = preprocess(obs, env=args.env).unsqueeze(0)
            obs_stack = torch.cat((obs_stack[:, 1:, ...], torch.tensor(obs, dtype=torch.float32).unsqueeze(1)), dim=1).to(device)

            #GJURT
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            old_obs_tensor = torch.tensor(old_obs_stack, dtype=torch.float32)
            action_tensor = torch.tensor([action], dtype=torch.int64)
            obs_tensor = torch.tensor(obs_stack, dtype=torch.float32)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            memory.push(old_obs_tensor, action_tensor, obs_tensor, reward_tensor, terminated)


            #GJURT?
            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if episode % env_config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            #GJURT
            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if episode % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')
            recorder.record_episode(episode, mean_return)

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
        
    # Close environment after training is completed.
    env.close()
    recorder.save()
