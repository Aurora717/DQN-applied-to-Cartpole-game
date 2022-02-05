# DQN

Deep Q-Network (DQN) is a deep reinforcement learning algorithm proposed by Deep Mind in 2015. DQN is a function approximation algorithm that replaces the 
Q-learning table by a neural network, it thus maps the input states of an  environment to the q-value action pairs using a parameterized function. What is 
attractive about DQNs is that they achieve stable learning via overcoming the deadly triad: Bootstrapping, Off Policy Learning, and function approximation. This is
done via implementing Experience-Replay and and Infrequent Updates.

Experience Replay: In deep reinforcement learning, the data used to train the network is not fixed like in supervised learning, it is created by interacting with 
the environment. Due to this interaction, the training states are correlated, as a result, the network may fall into a loop by being stuck in one area of the
state space, therefore, the scope of learning will be quite narrow. Experience replay is a method to circumvent this issue, previously visited states are stored in
a buffer, the training batches are then sampled randomly from this buffer. As a result, future training examples are not dependent on prior ones.

Infrequent Updates: Since DQN estimates qvalues, and the target function includes qvalue estimates. Any updates to the network changes the target function, as a result, the network may never converge as it is
essentially chasing its own tail. To overcome this, we create a delay between the network update and the target function. After a certain amount of N updates, we clone the original Q-network,
this cloned network is used to generate the targets, i.e, we use a different network as the target network. The use of two networks increases the stability of the learning process by setting a semi
”fixed” target to reach.

In the implementation, I apply DQN to a simple cartpole game. The cartpole is a game where we have a pole attached by a joint to a cart that moves on a 
friction-less track. The cart can move right and left on the track, the goal is to balance the pole on the cart.



