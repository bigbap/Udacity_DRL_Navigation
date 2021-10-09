
# Report for Navigation - Deep RL NanoDegree P1

The goal of this project is to build and train an agent that can solve Banana world.

The agent has to navigate an open world with yellow and blue banana's scattered around. A reward of 1 is given for walking over a yellow banana and -1 for walking over a blue one.

The world is solved once the agent has accumulated an average score of 13 over 100 episodes. An example of a trained agent can be seen below:

![trained agent](https://i.imgur.com/0JG7ud8.gif)

With the optimised implementation that I will outline in this report, I was able to train an agent to solve Banana world in just 381 episodes.

## 1. Implementation
The baseline agent uses a `fixed Q target network` and an `Experience Replay` buffer. The replay buffer stores experience tuples of `(s, r, a, s_, d, idx)`. `d` is a flag denoting episode termination, and `idx` is the index of the tuple in the buffer. The `idx` will be useful later when I discuss `Prioritised Experience Replay`.

In SL, we are able to train the agent towards a fixed target distribution that is known in advance. In RL, we don't have that luxury, and that is one of the first challenges we have to overcome. Because we are always training towards our next best guess and our best guess is always changing based on our previous training step, this causes a lot of variance when we are using the same online policy for making a prediction and estimating the value of that prediction.

Another problem we face is that each observation is highly correlated with the previous observation and action taken. If the agent is learning online in sequential order, the learned policy could become swayed by this correlation. This causes a lot of oscillation in the learning process and there is a possibility of converging towards the wrong policy.

We can solve both these problems by using a separate `Q network` for estimating the target values and by storing our experience tuples in an `experience replay buffer` that can be sampled from during learning. The paper [Human-level control through deep reinforcement learning](http://files.davidqiu.com//research/nature14236.pdf) is a good introduction to the DQN algorithm with fixed Q targets and experience replay.

### 1.1. Fixed Q Target Network
The idea here is that we have a separate DQN for estimating the target values in the learning step. This gives us some stability when learning.

At fixed intervals we update the target weights with the learned weights and that way we hope to move our target closer to the true value function. By learning offline from a separate DQN, the goal is to reduce some of the variance and converge faster.

In my implamentation, the separate `Neural Nets` are:

-  `dqn_local`. This is the predictor network
-  `dqn_target`. This is the target network

The target network is updated at regular intervals.

#### a. Neurals Nets
Bearing in mind that the environment has `state_space_n = 37` and `action_space_n = 4`, each neural net has the following layers:
 - `input layer = linear(state_space_n, 50)`
 - `hidden layer = linear(50, 50)`
 - `output layer = linear(50, action_space_n`

The input and hidden layers pass through a `leaky_relu` activation function in the forward pass.

The output layer doesn't pass through an activation function.

### 1.2. Experience Replay Buffer
The experience replay buffer is a list with a max length that stores experience tuples `(s, a, r, s_)`. At each time step, prediction is done using the Q network and the resulting experience tuple is stored in the replay buffer.

At regular intervals `k`, a batch of experience tuples is randomly sampled from the buffer and used to train the Q network.

Once the buffer is full, new experience tuples are added to the beginning of the list, replacing the old tuples that were there.

**(1.2.1)**

```
1. 	initialize max length N, and position H = -1
2. 	initialize buffer M = empty list
3. 	initialize batch size B

4. 	add E -> 
5. 		H 			= (H + 1) mod N
6.		M 			-> add experience E into M at position H

7. 	sample ->
8.		initialize minibatch MB = empty list

9.		for b = 1, B do
10.			MB 		-> sample randomly from M and append to MB
	
11.		return MB
```

### 1.3. Hyper-parameters, Loss Function and Optimiser
All agents were trained with the following hyper-parameters:
-  `gamma = 1` 			discount factor
-  `lr = 0.001` 		learning rate
-  `ep_start = 1` 		exploration start value
-  `ep_min = 0.01`		exploration min value
-  `ep_decay = 0.99`	exploration decay modifier
-  `k = 4` 				learning interval

The error function was `mean squared error` for all agents and the optimizer algorithm was `Adam`.

### 1.4. Baseline DQN Algorithm
Pseudocode for the baseline DQN algorithm is as follows:

**(1.4.1)**

```
1.	initialize experience replay D with capacity N
2.	initialize DQN and DQN_ with random weights

3.	initialize k, gamma, e = ep_start, lr

4.	for episode 1, M do
5.		s 				-> get initial state from environment
6.		d 				= false

7.		while not d
8.			a 			-> with probability e select random a, otherwise max(DQN(s))
9.			r, s_, d 		-> get (r, s_, d) from environment after executing a
10.			D 			-> store transition tuple (s, a, r, s_, d) in D
			
11.			every k
12.				s, a, r, s_ 	-> sample random minibatch of transitions from D
13.				prediction 	-> predict s value - DQN(s)[a]
14.				target 		-> get traget value - r + (gamma * max(DQN_(s_)) * (1 - d))

15.				error 		-> calculate mean squared error - MSE(prediction, target)
16.				update DQN	-> pass gradients w.r.t error through Adam optimizer
17.				DQN_ 		-> perform a soft update of DQN_ from DQN
```

The baseline agent was able to solve the environment in 1831 episodes.

**(1.4.2)**

![DQN](https://i.imgur.com/yBVutQ1.png)

### 1.5. Double Q-Learning
The problem with using the `max Q values` from the target network is that any target approximation errors will be positively biased. Since the target Q network is also initialized with random weights, it is basically guaranteed that there will be approximation errors.

To reduce this bias, we can implement a `Double Q-Learning` algorithm for target approximation. A detailed explanation of `Double Q-Learning` can be found in the paper [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461).

The implementation I used for a `DDQN` is the same as in **(1.4.1)**. The only difference is on line 15:

**(1.5.1)**

```
14.				prediction 	-> predict s value - DQN(s)[a]
15.				target 		-> get traget value - r + (gamma * DQN_(_s)[max(DQN(s_)] * (1 - d))

16.				loss 		-> calculate mean squared error - MSE(prediction, target)
```

Instead of using `max(DQN_(s_))` to approximate the target value, I use `DQN_(_s)[max(DQN(s_)]`. First, the local Q network `DQN` is used to approximate the value of `s_` when taking `max(a)`, then the target Q network `DQN_` is used to evaluate `a` that was chosen by `DQN`. Hence the name `Double Q-Learning`. Using this algorithm has the effect of smoothing the bias introduced by errors in the approximation function.

The DDQN agent with uniform sampling from the experience replay was able to solve the environment in 634 episodes.

**(1.5.2)**

![DDQN](https://i.imgur.com/QlacYEv.png)

### 1.6. Prioritised Experience Replay
There is a major problem with uniformly sampling from the experience replay. Most of the experiences might not be very informative. For example, if you are wandering around a grid world and not collecting any rewards, there isn't much to learn from that. It might take a while before a useful experience is sampled and in the meantime the agent is learning from a bunch of relatively useles experiences.

If we could prioritize experiences by how useful they are, that might speed up learning considerably. One obvious way that we could prioritize experiences is by how much the agent could learn from them. The `td error` is an obvious proxy for how much an agent can learn from an experience.

1. Along with the experience tuple, we will also store the priority `p = |td|` in the `experience replay` buffer.
2. If we sample greedily with respect to `p`, we run the risk of overfitting to a small subset of experiences. To mitigate this, we introduce a hyper-parameter alpha `a -> [0...1]` to determine how much weight we will give to the priority, with `a = 0` being equal to uniform sampling.
3. To avoid experiences with `td ≈ 0` from never getting sampled, we add a small constant epsillon `e` to the td error.

The final priority `p` that we store is `(|td| + e) ** a`.

#### a. Importance Sampling
By prioritizing certain experiences over others, we introduce bias into the learning process. This is because the expectation of the data distribution changes. We can correct this by using importance-sampling `IS` weights.

1. First, I calculate the sample probability `P = p(e) / sum(all p)`, where `e` is the experience and `p` is the priority.
2. The `IS` weight is calculated with `N * P`, where `N` is the total number of experiences in the buffer.
3. If we want to introduce some bias back into the mix, we can add a new hyper-parameter beta `b`. By raising `IS` to the power of `b`, we can control how much bias we want to have, where `b = 1` fully compensates for the bias introduced by the priority experience replay.

It makes sense that we might want more bias in the beginning of training since that's when we have the most to learn. And we would want to slowly temper the bias as we get closer to convergence. For this reason, I added a hyper-paramter `bi` for incrementing beta at each learning step. With `b = max(b + bi, 1)` the value of `b` will not go above 1.

Finally, the `IS` weights are normalized by `IS / max(IS)`. This ensures that we only scale the update downwards.

The final implementation can be seen below:

**(1.6.1)**

```
1. 	initialize max length N, position H = -1, max priority MP = 10000, beta b and beta increment bi
2. 	initialize buffer M = empty list
3. 	initialize batch size B

4. 	add E -> 
5. 		H 			= (H + 1) mod N
		E			-> add initial priority = MP to E. This ensures that each experience will be sampled
6.		M 			-> add experience E into M at position H

7. 	sample ->
8.		initialize minibatch MB = empty list
9.		b			= max(b + bi, 1) increment beta

10.		for b = 1, B do
11.			e		= sample experience by priority
12.			P		= p(e) / sum(all p)
13.			IS		= (N * P) ** -b
14.			IS		= IS / max(IS) normalize weights
15.			MB 		-> append (e, IS) to MB
	
16.		return MB
```

For a more detailed explanation of prioritized experience replay, consult the paper [Prioritised Experience Replay](https://arxiv.org/abs/1511.05952).

With PER, the DQN and DDQN agents were able to solve the environment in 797 and 381 episodes respectively.

**(1.6.1)**

![DQN/PER](https://i.imgur.com/XNVKgNn.png)

**(1.6.2)**

![DDQN/PER](https://i.imgur.com/EhZ2Ovv.png)

## 2. Side By Side
Below is a side by side comparison of each agent learning to solve Banana world.

**(2.1)**

![Comparison](https://i.imgur.com/wxRx8RN.png)

## 3. Ideas for the Future

- implement duelling DQN
