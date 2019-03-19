"""Reference Flux implementation of the policy gradient algorithm."""

using Flux, CuArrays, Distributions, Statistics, PyCall
using Flux: params
using Flux.Tracker: TrackedReal, gradient, update!
gym = pyimport("gym")


"""Given an environment, construct a policy."""
function construct_policy(env::PyObject, device::Function)
    state_space = env.observation_space.shape[1]
    hidden_units = 20
    action_space = env.action_space.n
    Chain(Dense(state_space, hidden_units, relu),
          Dense(hidden_units, action_space), softmax
          ) |> device
end


"""
Select an action using the policy given the curren state.

Return the action and the log probability of taking that action.
"""
function select_action(policy::Chain, state::Vector{Float64}, device::Function)
    probs = policy(device(state))
    m = Categorical(cpu(probs).data)
    action = rand(m)
    action, log.(probs)[action]
end


"""
Play a single episode using the policy in the environment.

Returns the rewards and log probablities of the action taken
at each timestep.
"""
function play_episode(policy::Chain, env::PyObject, device::Function;
                      render=false)
    rewards = Float64[]
    log_probs = TrackedReal[]
    state = env.reset()
    done = false
    while !done
        action, log_prob = select_action(policy, state, device)
        state, reward, done, _ = env.step(action - 1)
        push!(rewards, reward)
        push!(log_probs, log_prob)
        render && env.render()
    end
    rewards, log_probs
end


"""Discount the rewards by γ."""
function discount(rewards::Vector{Float64}; γ::Float64)
    discounted = similar(rewards)
    running_sum = 0
    for i in length(rewards):-1:1
        running_sum = rewards[i] + γ * running_sum
        discounted[i] = running_sum
    end
    discounted
end


"""Normalize rewards to have mean of 0 and std deviation of 1."""
normalize(rewards::Vector{Float64}) =
    (rewards .- mean(rewards)) / (std(rewards) + eps(Float64))


"""Improve the policy utilizing the policy gradient algorithm."""
improve_policy(policy::Chain, env::PyObject, device::Function,
               optimizer::ADAM; episodes=100) =
    for _ in 1:episodes
        rewards, log_probs = play_episode(policy, env, device)
        returns = normalize(discount(rewards, γ=0.99))
        θ = params(policy)
        policy_gradient = gradient(θ) do
            sum(-log_prob * r for (log_prob, r) in zip(log_probs, returns))
        end
        update!(optimizer, θ, policy_gradient)
    end


env = gym.make("CartPole-v0")

device = gpu

policy = construct_policy(env, device)

optimizer = ADAM(1e-2)

sum(play_episode(policy, env, device, render=true)[1])

improve_policy(policy, env, device, optimizer, episodes=100)

env.close()
