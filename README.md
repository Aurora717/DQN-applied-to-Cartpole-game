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

![poster](https://user-images.githubusercontent.com/49812606/152653378-2a81df8a-c8c8-429c-bfd7-5ceb8cfd90f3.jpg)

The DQN implementation consists of 3 primary components, the gaming environment as mentioned above, the agent, and the replay buffer. 
 1. Replay-Memory: This block consists of 2 functions: a Memory function that stores experiences, and a Replay function that take a random sample of the stored experiences to be used as the training batch. 
 2. Agent: This block primarily consists of a deep neural network and  a policy. The function of the agent is to optimize the Qvalue prediction (target policy) using the network mentioned above and then chooses an action given the state via the policy.

Results:

![average_reward](https://user-images.githubusercontent.com/49812606/152653567-55b79c4b-32fd-4681-bcad-38e5008c0b57.png)
![loss](https://user-images.githubusercontent.com/49812606/152653568-852e7646-1309-4fd4-84e4-1cc195d6892a.png)
