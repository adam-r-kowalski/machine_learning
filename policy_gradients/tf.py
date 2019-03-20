"""Reference TensorFlow implementation of the policy gradient algorithm."""

# %% imports
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers, Sequential, activations, optimizers, Model
from tensorflow_probability import distributions as tfd
import numpy as np
import gym
from typing import Tuple, List


# %% definitions
def construct_policy(env: gym.Env) -> Model:
    """Given an environment, construct a policy."""
    hidden_units = 20
    action_space = env.action_space.n
    return Sequential([
        layers.Dense(hidden_units, activations.relu),
        layers.Dense(action_space, activations.softmax)
        ])


def select_action(policy: Model, state: np.ndarray) -> Tuple[int, Tensor]:
    """
    Select an action using the policy given the curren state.

    Return the action and the log probability of taking that action.
    """
    probs = policy(tf.expand_dims(state, axis=0))
    m = tfd.Categorical(probs=probs)
    action = m.sample()
    return int(action), m.log_prob(action)


def play_episode(policy: Model, env: gym.Env,
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
        action, log_prob = select_action(policy, state)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        if render:
            env.render()
    return tf.constant(rewards, dtype=tf.float64), tf.stack(log_probs)


def discount(rewards: Tensor, gamma: float) -> Tensor:
    """Discount the rewards by gamma."""
    discounted = np.zeros_like(rewards)
    running_sum = 0
    for i in reversed(range(len(rewards))):
        running_sum = rewards[i] + gamma * running_sum
        discounted[i] = running_sum
    return tf.constant(discounted)


def normalize(rewards: Tensor) -> Tensor:
    """Normalize rewards to have mean of 0 and std deviation of 1."""
    eps = np.finfo(np.float32).eps.item()
    mean, variance = tf.nn.moments(rewards, axes=0)
    return (rewards - mean) / (tf.sqrt(variance) + eps)


def improve_policy(policy: Model, env: gym.Env,
                   optimizer: optimizers.Optimizer, episodes=100):
    """Improve the policy utilizing the policy gradient algorithm."""
    for _ in range(episodes):
        with tf.GradientTape() as tape:
            rewards, log_probs = play_episode(policy, env)
            returns = normalize(discount(rewards, gamma=0.99))
            policy_loss = sum(-tf.squeeze(log_probs) * returns)
        vars = policy.trainable_variables
        grads = tape.gradient(policy_loss, vars)
        optimizer.apply_gradients(zip(grads, vars))


# %% entry point
env = gym.make("CartPole-v0")

policy = construct_policy(env)

optimizer = optimizers.Adam(learning_rate=1e-2)

sum(play_episode(policy, env, render=True)[0])

improve_policy(policy, env, optimizer, episodes=100)

env.close()
