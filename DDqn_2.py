import random
from collections import deque
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam 
from keras.optimizers import RMSprop
import gym
import time
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import animation
import moviepy.editor as mp



class ReplayMemory:
   

    def __init__(self):
        self.experiences = deque(maxlen=1000000)


    def Memory_func(self, state, next_state, reward, action, terminal):

        self.experiences.append((state, next_state, reward, action,terminal))

    def Replay_func(self):
     
        batch_size = min(256, len(self.experiences))
        samples = random.sample(self.experiences, batch_size)
        
        states,next_states, rewards, actions,terminals =  map(list,zip(*samples))
        
    
        return np.array(states), np.array(next_states), np.array(
            actions), np.array(rewards), np.array(terminals)




class Agent:

    def __init__(self):
        
        self.gamma = 0.95
        self.learning_rate = 0.001
   
        self.epsilon= 0.1
        #self.epsilon_end = 0.1
        #self.epsilon_decay = 0.001
        
        self.Qnet = self.DQN_Model()
        self.target_net = self.DQN_Model()
        
        
    
    @staticmethod
    def DQN_Model():
        
        model = Sequential()
        model.add(Dense(512, input_dim=4, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(2, activation='linear', kernel_initializer='he_uniform'))
        model.compile(optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),loss='mse')
        return model
   
        
    def policy(self, state):
       
        #rate = self.exploration_rate()
       
        if np.random.random() < self.epsilon:
            return np.random.choice([0,1])
        else:
            
            inputs = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
            possible_actions = self.Qnet(inputs)
            best_action = np.argmax(possible_actions.numpy()[0], axis=0)
            return best_action


    def update(self):
        self.target_net.set_weights(self.Qnet.get_weights())
        
        
    def transition(self , target_qvalues_r, max_qvalues):
        
        target_qvalues_r += self.gamma * max_qvalues
            
        return target_qvalues_r
    
    
        
    def save(self):
        self.Qnet.save("Qnet.h5")
        
        
    def load(self, name):
        
        self.Qnet = load_model(name)
        
    def train_network(self, batch):
       
        states, next_states, actions, rewards, terminals = batch
        
        states_qvalues = self.Qnet(states).numpy()
        target_qvalues = np.copy(states_qvalues)
        next_states_qvalues = self.target_net(next_states).numpy()
        max_qvalues = np.amax(next_states_qvalues, axis=1)
        
        for i in range(states.shape[0]):
            target_qvalues_r = rewards[i]
            
            if not terminals[i]:
                target_qvalues_r =  self.transition(target_qvalues_r, max_qvalues[i])
                
            target_qvalues[i][actions[i]] = target_qvalues_r
            
        training_history = self.Qnet.fit(x=states, y=target_qvalues, verbose=0)
        loss = training_history.history['loss']
        return loss



def plot(result, file_name, title):
    
    episodes = np.arange(len(result))
    plt.figure(2)
    plt.clf()
    plt.title('Trained Model ')
    plt.xlabel('Episode')
    plt.ylabel(title)
    plt.plot(episodes,result )
    plt.savefig(file_name)
    
    
    
def average_reward(env, agent):
    
    reward_t = []
    #reward_t = 0.0
    episodes_to_play = 10
    for i in range(episodes_to_play):
        state = env.reset()
        terminal = False
        reward_eps = 0.0
        while not terminal:
            action = agent.policy(state)
            next_state, reward, terminal, _ = env.step(action)
            reward_eps += reward
            state = next_state
        reward_t.append(reward_eps)
        #reward_t+=reward_eps
   # average_reward = reward_t / episodes_to_play
    average_reward = np.mean(reward_t)
    return average_reward


def create_experiences(env, agent, replay, amount):
   
    for i in range(amount):
        state = env.reset()
        terminal = False
        while not terminal:
            action = agent.policy(state)
            next_state, reward, terminal, _ = env.step(action)
            if terminal:
                reward = -1.0
            replay.Memory_func(state, next_state,
                                             reward, action, terminal)
            state = next_state


def run_model():
   
    start = time.time()
    Episodes=2500
    agent = Agent()
    replay = ReplayMemory()
    env = gym.make('CartPole-v1')
    loss_list,reward_list = deque([]),deque([])
 
    
    create_experiences(env, agent, replay, 100)
        
    for episode in range(Episodes):
        
        create_experiences(env, agent, replay,1)
        experience_batch = replay.Replay_func()

        loss = agent.train_network(experience_batch)
        avg_reward = average_reward(env, agent)
        
        print('Episode: ',episode,"/",Episodes, 'Average Reward : ' ,avg_reward)
              
        
        loss_list.append(loss)
        reward_list.append(avg_reward)
        
        #env.render()
        if episode % 20 == 0:
            agent.update()
            
        if episode > 750:
            
                if avg_reward == 500.0:
                    agent.save()
                    break 
    env.close()
    
    plot(reward_list, 'average_reward', 'Average Reward')
    plot(loss_list, 'loss', 'Training Loss')
    print('Total time taken to train network is ', (time.time() - start)/60, ' minutes')


if __name__ == "__main__":
    
    run_model()

    
