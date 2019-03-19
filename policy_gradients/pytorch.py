"""Reference PyTorch implementation of the policy gradient algorithm."""

# %% imports
import torch
from torch import Tensor
from torch.distributions import Categorical
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from typing import Tuple, List


# %% definitions
def construct_policy(env: gym.Env, device: torch.device) -> nn.Module:
    """Given an environment, construct a policy."""
    state_space = env.observation_space.shape[0]
    hidden_units = 20
    action_space = env.action_space.n
    return nn.Sequential(
        nn.Linear(state_space, hidden_units), nn.ReLU(),
        nn.Linear(hidden_units, action_space), nn.Softmax(dim=1)
        ).to(device)


def select_action(policy: nn.Module, state: np.ndarray,
                  device: torch.device) -> Tuple[int, Tensor]:
    """
    Select an action using the policy given the curren state.

    Return the action and the log probability of taking that action.
    """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)


def play_episode(policy: nn.Module, env: gym.Env, device: torch.device,
                 render=False) -> Tuple[Tensor, Tensor]:
    """
    Play a single episode using the policy in the environment.

    Returns the rewards and log probablities of the action taken
    at each timestep.
    """
    rewards: List[float] = []
    log_probs: List[Tensor] = []
    state = env.reset()
    done = False
    while not done:
        action, log_prob = select_action(policy, state, device)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        if render:
            env.render()
    return torch.tensor(rewards), torch.stack(log_probs)


def discount(rewards: Tensor, gamma: float) -> Tensor:
    """Discount the rewards by gamma."""
    discounted = torch.zeros_like(rewards)
    running_sum = 0
    for i in reversed(range(len(rewards))):
        running_sum = rewards[i] + gamma * running_sum
        discounted[i] = running_sum
    return discounted


def normalize(rewards: Tensor) -> Tensor:
    """Normalize rewards to have mean of 0 and std deviation of 1."""
    eps = np.finfo(np.float32).eps.item()
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def improve_policy(policy: nn.Module, env: gym.Env, device: torch.device,
                   optimizer: optim.Optimizer, episodes=100):
    """Improve the policy utilizing the policy gradient algorithm."""
    for _ in range(episodes):
        rewards, log_probs = play_episode(policy, env, device)
        returns = normalize(discount(rewards, gamma=0.99))
        optimizer.zero_grad()
        (-log_probs.squeeze() * returns.to(device)).sum().backward()
        optimizer.step()


# %% entry point
env = gym.make("CartPole-v0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = construct_policy(env, device)

optimizer = optim.Adam(policy.parameters(), lr=1e-2)

sum(play_episode(policy, env, device, render=True)[0])

improve_policy(policy, env, device, optimizer, episodes=100)

env.close()
