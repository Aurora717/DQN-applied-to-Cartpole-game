import gym
import time
import matplotlib.pyplot as plt
from matplotlib import animation
import moviepy.editor as mp
from DDqn_2 import ReplayMemory, Agent
import cv2
import numpy as np
import glob





def frames_to_gif(frames, path, filename):

    def anime(i):
       
        show.set_data(frames[i])
    
    print("Turning Simulation into a gif and saving it....")
    print("this may take a few minutes....")
    plt.figure(figsize=(frames[0].shape[1] / 80.0, frames[0].shape[0] / 80.0), dpi=70)
    show = plt.imshow(frames[0])
    plt.axis('off')
    
    gif = animation.FuncAnimation(plt.gcf(), anime, frames = len(frames), interval=50)
    gif.save(path + filename, writer='Pillow', fps=40)
    print("Simulation saved as a gif")

def gif_to_movie(gif_name):
    
    print("Turning gif to a movie")
    movie = mp.VideoFileClip(gif_name)
    movie.write_videofile("CartPole_dqn.mp4")
    print('Movie saved')




  	

def test_model():
   
    agent = Agent()
    agent.load('Qnet.h5')
    Episodes = 10
    env = gym.make('CartPole-v1')
    reward_t = 0.0
    frames = []
    for i in range(Episodes):
        state = env.reset()
        terminal = False
        reward_eps = 0.0
        while not terminal:
            #env.render()
            frames.append(env.render(mode="rgb_array"))
            
            action = agent.policy(state)
            next_state, reward, terminal, _ = env.step(action)
            reward_eps += reward
            
            state = next_state
        print("Episode ", i+1 , "Reward is :, ", reward_eps)
        #reward_t.append(reward_eps)
        reward_t+= reward_eps
    average_reward = reward_t / Episodes
    #average_reward = np.mean(reward_t)
    print("Average_Reward : ", average_reward)
    env.close()


    frames_to_gif(frames,  './', 'cartpole_dqn.gif')
    gif_to_movie(gif_name = 'cartpole_dqn.gif')


if __name__ == "__main__":
    
	test_model()
