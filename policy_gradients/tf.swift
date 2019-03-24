import Python
import TensorFlow

let gym = Python.import("gym")
let env = gym.make("CartPole-v0")

func numpyToTensor(_ numpy: PythonObject) -> Tensor<Float> {
  return Tensor<Float>(Tensor<Double>(numpy: numpy)!).reshaped(to: [1, 4])
}

struct Policy : Layer {
  var l1, l2: Dense<Float>
  
  init(_ env: PythonObject) {
    let stateSpace = Int(env.observation_space.shape[0])!
    let hiddenUnits = 20
    let actionSpace = Int(env.action_space.n)!
    l1 = Dense<Float>(
       inputSize: stateSpace, outputSize: hiddenUnits, activation: relu)
    l2 = Dense<Float>(
       inputSize: hiddenUnits, outputSize: actionSpace, activation: softmax)
  }
  
  @differentiable
  func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
    return l2.applied(to: l1.applied(to: input, in: context), in: context)
  }
}

func choice(_ probs: Tensor<Float>) -> Int32 {
  var roll = Tensor<Float>(randomUniform: [1])[0]
  for i in (0..<probs.shape[1]).reversed() {
    let prob = probs[0][i]
    if roll <= prob {
      return i
    }
    else {
      roll -= prob
    }
  }
  return 0
}

func selectAction(_ policy: Policy, _ state: Tensor<Float>,
                  in context: Context) -> (Int32, Tensor<Float>) {
  let probs = policy.applied(to: state, in: context)
  let action = Int32(choice(probs))
  return (action, log(probs[0][action]))
}

func playEpisode(_ policy: Policy, _ env: PythonObject,
                 in context: Context, render shouldRender: Bool = false)
                 -> (Tensor<Float>, Tensor<Float>) {
  var rewards: [Float] = []
  var logProbs: [Tensor<Float>] = []
  var state = numpyToTensor(env.reset())
  var done = false
  while !done {
    let (action, logProb) = selectAction(policy, state, in: context)
    let (nextState, reward, isDone, _) = env.step(action).tuple4
    rewards.append(Float(reward)!)
    logProbs.append(logProb)
    state = numpyToTensor(nextState)
    done = Bool(isDone)!
  }
  return (Tensor<Float>(rewards), Tensor<Float>(logProbs))
}

func discount(_ rewards: Tensor<Float>, gamma: Float) -> Tensor<Float> {
  var discounted = Tensor<Float>(zeros: rewards.shape)
  var running_sum = Float(0.0)
  for i in (0..<rewards.shape[0]).reversed() {
    running_sum = Float(rewards[i])! + gamma * running_sum
    discounted[i] = Tensor<Float>(running_sum)
  }
  return discounted
}

func std(_ x: Tensor<Float>) -> Tensor<Float> {
  let mean = x.mean()
  return sqrt(pow(x - mean, 2).sum() / Float(x.shape[0]))
}

func normalize(_ rewards: Tensor<Float>) -> Tensor<Float> {
  return (rewards - rewards.mean()) / std(rewards)
}

func improvePolicy(_ policy: inout Policy, _ env: PythonObject,
                   _ optimizer: Adam<Policy, Float>,
                   in context: Context, episodes: Int) {
  for _ in 1...episodes {
    let (rewards, logProbs) = playEpisode(policy, env, in: context)
    let returns = normalize(discount(rewards, gamma: 0.99))
    let gradients = gradient(at: policy) { model -> Tensor<Float> in
      return (-logProbs * returns).sum()
    }
    optimizer.update(&policy.allDifferentiableVariables, along: gradients)
  }
}

var policy = Policy(env)
let optimizer = Adam<Policy, Float>(learningRate: 0.01)
let context = Context(learningPhase: .training)

let (rewards, _) = playEpisode(policy, env, in: context)
rewards.sum()

improvePolicy(&policy, env, optimizer, in: context, episodes: 100)

let (rewards, _) = playEpisode(policy, env, in: context)
rewards.sum()

