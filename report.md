# Continuous control with DDPG network

## objectives
<br>
The goal of the project is to train 20 robotic arms within the [Udacity Reacher environment](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).

<br>

### Environment
<br>

The 20 robotic arms are able to move in 3d space by providing torque in each of the 4 directions defined in the environment. 

<br>

### goals
<br>
to maintain contact with all the green spheres for as long as possible. 

### rewards
<br>
a reward of +0.1 is provided for maintaining contact for 1 time step.

<br>

### solved game criteria
<br>
to obtain a higher than 30 average score over 100 episodes

<br>

### action spaces
<br>

4 action spaces corresponds to torque in 4 directions

<br>

### state space
<br>

33 state dimensions corresponds to location of all objects


## Model


### DDPG model

#### Actor
ANN with 3 layers size of
input = 33 states
fp1 = 256
fp2 = 128
output = 4

#### Critic
ANN with 3 layers

### Hyperparameters


![](http://file.hatari.cc/yqH1V/hyperparam_multi_agent.png)


#### Noise Hyperparameteres
```
SIGMA = 0.05          # Ornstein-Uhlenbeck noise parameter
EPSILON = 1.0           # explore->exploit noise process added to act step
EPS_DECAY = 1e-6    # decay rate for noise process
```
#### learning
```
LR_ACTOR = 1e-3 #learning rate for actor
LR_CRITIC = 1e-3 #learning rate for critic
WEIGHT_DECAY = 0
GAMMA = 0.99
TAU = 1e-3
BATCH_SIZE = 1024
BUFFER_SIZE = 1e6

NUM_UPDATES = 4 #how many passes of experience replay to learn from
UPDATE_EVERY = 20 # learning interval
```


## Results

### Baseline 

A random choice agent choose action at random.
```
env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
```

The agent obtain approximately 0.1 score in average.




### Training performance 

The agent first reaches the score of 30 at episode 56, the score increases after that. At episode 140, the average score over the past 100 episodes is over 30 points.


![](https://nicrl.s3.amazonaws.com/results/training_plot28_20%3A31.png)
<BR><i>average scores of ddpg_network using 4 updates per 20 episodes</i>


## Further Improvements

1. Optimization of noise parameters
2. Optimization of the model.py parameters. 
* The actor and critic network could make use of RNN instead of CNN
* a deeper network can be used 
* a larger dense network can be used
3. testing different optimizers and parameters
* for this experiment, Adam optimizer is use while training. However many other algorithms can be used including genetics, ADAGrad, RMS and many others. This will affect the training performance of the netowork.

