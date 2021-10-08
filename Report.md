
# Report for Navigation - Deep RL NanoDegree P1

The goal of this project is to build and train an agent that can solve Banana world.

The agent has to navigate an open world with yellow and blue banana's scattered around. A reward of 1 is given for walking over a yellow banana and -1 for walking over a blue one.

The world is solved once the agent has accumulated an average score of 13 over 100 episodes. An example of a trained agent can be seen below:

![trained agent](https://i.imgur.com/0JG7ud8.gif)

With the optimised implementation that I will outline in this report, I was able to train an agent to solve Banana world in just 381 episodes.

## Implementation
The baseline agent has 2 separate `Neural Nets` and an `Experience Replay` buffer. The replay buffer stores experience tuples of `(s, r, a, s_, d, idx)`. `d` is a flag denoting episode termination, and `idx` is the index of the tuple in the buffer. The `idx` will be useful later when I discuss `Prioritised Experience Replay`.

The separate `Neural Nets` are:

-  `dqn_local`. This is the predictor network
-  `dqn_target`. This is the target network

#### Neurals Nets
Bearing in mind that the environment has `state_space_n = 37` and `action_space_n = 4`, each neural net has the following layers:
 - `input layer = linear(state_space_n, 50)`
 - `hidden layer = linear(50, 50)`
 - `output layer = linear(50, action_space_n`

The input layer and the hidden layer pass through a `leaky_relu()` activation function in the forward pass.

The output layer doesn't pass through an activation function.

#### Training Hyper-parameters
All agents were trained with the following hyper-parameters:
-  `gamma = 1`
-  `lr = 0.001`
-  `ep_start = 1`
-  `ep_min = 0.01`
-  `ep_decay = 0.99`
-  `learn_every = 4` learning interval

### Baseline DQN
Pseudocode for the baseline DQN algorithm is as follows:
```
U = experience replay
DQN = DQN local
DQN_ = DQN target

k = learning interval
gamma = discount factor

s, d -> env_initial()
while not d
	a -> e_greedy(DQN(s))
	r, s_, d -> env(s, a)
	U -> append(<s, a, r, s_, d>)
	
	every k
		s, a, r, s_ -> sample(U)
		prediction -> DQN(s)[a]
		target -> r + (gamma * max(DQN_(s_)) * (1 - d))

		loss -> MSE(prediction, target)
		optimise DQN
		DQN_ -> soft_update(DQN)
```

The baseline agent was able to solve the environment in 1831 episodes.

![DQN](https://i.imgur.com/yBVutQ1.png)

### Double DQN
The first optimisation I made was to implement a Double DQN agent. Instead of using `max(DQN_(s_))` as the learning target, with a DDQN I use `DQN_(_s)[max(DQN(s_)]`.

TODO: explain why this is an improvement and reference [Double DQN paper](https://arxiv.org/abs/1509.06461)

The DDQN agent with uniform sampling from the experience replay was able to solve the environment in 634 episodes.

![DDQN](https://i.imgur.com/QlacYEv.png)


### Prioritised Experience Replay
TODO: explain why this is an improvement and reference [Prioritised Experience Replay Paper](https://arxiv.org/abs/1511.05952)

With PER, the DQN and DDQN agents were able to solve the environment in 797 and 381 episodes respectively.

![DQN/PER](https://i.imgur.com/XNVKgNn.png)

![DDQN/PER](https://i.imgur.com/EhZ2Ovv.png)

## Ideas for the Future

- implement duelling DQN