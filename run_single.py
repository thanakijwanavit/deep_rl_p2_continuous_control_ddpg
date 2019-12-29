from collections import deque
from importlib import reload
import ddpg_agents
from ddpg_agents import Agent
import torch
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
import numpy as np
import pandas as pd
import datetime


#env = UnityEnvironment(file_name='./Reacher_single/Reacher_Linux_NoVis/Reacher.x86_64')
env = UnityEnvironment(file_name='./Reacher_single/Reacher_Linux/Reacher.x86_64')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
print('state size is {} action size is {}'.format(state_size,action_size))

def ddpg(n_episodes=1000, max_t=300, print_every=1, num_updates = 10):
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2, num_updates = num_updates)
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            if i_episode == 1 and t == 1: print('training started successfully')
            actions = agent.act(states,add_noise=True)
#             print('next action is {}'.format(action))
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            if i_episode == 1 and t == 1: print('variables are rewards: {} actions: {}'.format(rewards,actions))
#             print(done)
#             next_state, reward, done, _ = env.step(action)
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done , t)
            states = next_states
#             print('reward is {}'.format(rewards))
            score += rewards
            if t % 10 == 0:
                print('episode {} action {}'.format(i_episode, t))
            if np.any(done):
                print('completed episode {} at t of {}'.format(i_episode,t))
#                 print(done)
                break
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
#         print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores

if __name__=='__main__':
    for i_update in [8,7,6]:
        scores = ddpg(300,1000,num_updates = i_update)
        dt = datetime.datetime.now()
        time_for_name = dt.strftime("%d_%H:%M")
        df = pd.DataFrame({'scores': scores })
        df.to_csv('results/training_result{}update{}.csv'.format(time_for_name,i_update))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig('results/training_plot{}update{}.png'.format(time_for_name,i_update))
