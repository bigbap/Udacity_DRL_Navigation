
# Report for Navigation - Deep RL NanoDegree P1

The goal of this project is to build and train an agent that can solve Banana world.

The agent has to navigate an open world with yellow and blue banana's scattered around. A reward of 1 is given for walking over a yellow banana and -1 for walking over a blue one.

The world is solved once the agent has accumulated an average score of 13 over 100 episodes. An example of a trained agent can be seen below:

![trained agent](https://i.imgur.com/0JG7ud8.gif)

With the optimised implementation that I will outline in this report, I was able to train an agent to solve Banana world in just 381 episodes.

## Implementation
The baseline agent uses a `fixed Q target network` and an `Experience Replay` buffer. The replay buffer stores experience tuples of `(s, r, a, s_, d, idx)`. `d` is a flag denoting episode termination, and `idx` is the index of the tuple in the buffer. The `idx` will be useful later when I discuss `Prioritised Experience Replay`.

In SL, we are able to train the agent towards a fixed target distribution that is known in advance. In RL, we don't have that luxury, and that is one of the first challenges we have to overcome. Because we are always training towards our next best guess and our best guess is always changing based on our previous training step, this causes a lot of variance when we are using the same online policy for making a prediction and estimating the value of that prediction.

Another problem we face is that each observation is highly correlated with the previous observation and action taken. If the agent is learning online in sequential order, the learned policy could become swayed by this correlation. This causes a lot of oscillation in the learning process and there is a possibility of converging towards the wrong policy.

We can solve both these problems by using a separate `fixed Q network` for estimating the target values and by storing our experience tuples in a `replay buffer` that can be sampled from during learning.

### Fixed Q Target Network
The idea here is that we have a separate DQN for estimating the target values in the learning step. This gives us some stability when learning.

At fixed intervals we update the target weights with the learned weights and that way we hope to move our target closer to the true value function. By learning offline from a separate DQN, the goal is to reduce some of the variance and converge faster.

In my implamentation, the separate `Neural Nets` are:

-  `dqn_local`. This is the predictor network
-  `dqn_target`. This is the target network

The target network is updated at regular intervals.

#### Neurals Nets
Bearing in mind that the environment has `state_space_n = 37` and `action_space_n = 4`, each neural net has the following layers:
 - `input layer = linear(state_space_n, 50)`
 - `hidden layer = linear(50, 50)`
 - `output layer = linear(50, action_space_n`

The input and hidden layers pass through a `leaky_relu` activation function in the forward pass.

The output layer doesn't pass through an activation function.

### Experience Replay Buffer
The experience replay buffer is a list with a max length that stores experience tuples `(s, a, r, s_)`. At each time step, prediction is done using the Q network and the resulting experience tuple is stored in the replay buffer.

At regular intervals `(learning interval k)`, a batch of experience tuples is randomly sampled from the buffer and used to train the Q network.

Once the buffer is full, new experience tuples are added to the beginning of the list, replacing the old tuples that were there.

```
1. 	initialize max length N, and time step t = -1
2. 	initialize buffer M = empty list
3. 	initialize batch size B

4. 	add experience E -> 
5. 		t 			= (t + 1) mod N
6.		E 			-> into M at position t

7. 	sample ->
8.		initialize minibatch MB = empty list
9.		for b = 1, B do
10.			MB 		-> sample randomly from M and append to MB
	
11.		return MB
```

### Training Hyper-parameters
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
1.	initialize experience replay D with capacity N
2.	initialize DQN and DQN_ with random weights

3.	k 		= learning interval
4.	gamma 		= discount factor

5.	for episode 1, M do
6.		s 				-> get initial state from environment
7.		d 				= false
8.		while not d
9.			a 			-> with probability e select random a, otherwise max(DQN(s))
10.			r, s_, d 		-> get (r, s_, d) from environment after executing a
11.			D 			-> store transition tuple (s, a, r, s_, d) in D
			
12.			every k
13.				s, a, r, s_ 	-> sample random minibatch of transitions from D
14.				prediction 	-> predict s value - DQN(s)[a]
15.				target 		-> get traget value - r + (gamma * max(DQN_(s_)) * (1 - d))

16.				loss 		-> calculate mean squared error - MSE(prediction, target)
17.				perform gradient descent step on DQN
18.				DQN_ 		-> perform a soft update of DQN_ from DQN
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
