// Reference PyTorch implementation of the policy gradient algorithm.

#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace py = pybind11;
namespace nn = torch::nn;
namespace optim = torch::optim;
using Tensor = torch::Tensor;

// Convert a numpy array to a torch array.
auto numpy_to_torch(py::array &&numpy) -> Tensor {
  const auto shapes =
      std::vector<int64_t>{numpy.shape(), numpy.shape() + numpy.ndim()};
  return torch::from_blob(numpy.mutable_data(), torch::ArrayRef(shapes),
                          torch::kF64)
      .to(torch::kF32);
}

// Given an environment, construct a policy.
auto construct_policy(const py::object &env) -> nn::Sequential {

  const auto state_space =
      py::cast<int>(py::tuple(env.attr("observation_space").attr("shape"))[0]);
  const auto hidden_units = 20;
  const auto action_space = py::cast<int>(env.attr("action_space").attr("n"));
  return nn::Sequential(nn::Linear(state_space, hidden_units),
                        nn::Functional(torch::relu),
                        nn::Linear(hidden_units, action_space));
}

auto choice(const Tensor &probs) -> int {
  auto roll = torch::rand(1).item<float>();
  for (int i = 0; i < probs.size(0); ++i) {
    const auto prob = probs[i].item<float>();
    if (roll <= prob)
      return i;
    else
      roll -= prob;
  }
  throw("invalid probabilty tensor");
}

/*
  Select an action using the policy given the curren state.

  Return the action and the log probability of taking that action.
 */
auto select_action(nn::Sequential &policy, const Tensor &state)
    -> std::tuple<int, Tensor> {
  const auto probs = torch::softmax(policy->forward(state), /*dim=*/0);
  const auto action = choice(probs);
  return std::make_tuple(action, probs[action].log());
}

/*
  Play a single episode using the policy in the environment.

  Returns the rewards and log probablities of the action taken
  at each timestep.
 */
auto play_episode(nn::Sequential &policy, const py::object &env)
    -> std::tuple<Tensor, Tensor> {
  auto rewards = std::vector<float>{};
  auto log_probs = std::vector<Tensor>{};
  auto state = numpy_to_torch(env.attr("reset")());
  auto done = false;
  while (!done) {
    auto [action, log_prob] = select_action(policy, state);
    py::tuple data = env.attr("step")(action); /* state, reward, done, info */
    state = numpy_to_torch(data[0]);
    rewards.push_back(py::cast<float>(data[1]));
    done = py::cast<bool>(data[2]);
    log_probs.push_back(std::move(log_prob));
  }

  return std::make_tuple(torch::tensor(std::move(rewards)),
                         torch::stack(std::move(log_probs)));
}

// Discount the rewards by gamma.
auto discount(const Tensor &rewards, float gamma) -> Tensor {
  auto discounted = torch::zeros_like(rewards);
  auto running_sum = 0.0;
  for (int i = rewards.size(0) - 1; i >= 0; --i) {
    running_sum = rewards[i].item<float>() + gamma * running_sum;
    discounted[i] = running_sum;
  }
  return discounted;
}

// Normalize rewards to have mean of 0 and std deviation of 1.
auto normalize(const Tensor &rewards) -> Tensor {
  return (rewards - rewards.mean()) / rewards.std();
}

// Improve the policy utilizing the policy gradient algorithm.
auto improve_policy(nn::Sequential &policy, const py::object &env,
                    optim::Optimizer &optimizer, int episodes = 100) {
  for (int i = 0; i < episodes; ++i) {
    const auto [rewards, log_probs] = play_episode(policy, env);
    const auto returns = normalize(discount(rewards, /*gamma=*/0.99));
    optimizer.zero_grad();
    (-log_probs.squeeze() * returns).sum().backward();
    optimizer.step();
  }
}

auto main() -> int {
  const auto guard = py::scoped_interpreter{};

  const auto gym = py::module::import("gym");

  const auto env = gym.attr("make")("CartPole-v0");

  auto policy = construct_policy(env);

  auto state = numpy_to_torch(env.attr("reset")());

  auto optimizer = optim::Adam(policy->parameters(), /*lr=*/0.01);

  auto [rewards_before, _1] = play_episode(policy, env);

  std::cout << "rewards before training = "
            << rewards_before.sum().item<float>() << "\n";

  improve_policy(policy, env, optimizer, /*episodes=*/100);

  auto [rewards_after, _2] = play_episode(policy, env);

  std::cout << "rewards after training = " << rewards_after.sum().item<float>();

  return 0;
}
