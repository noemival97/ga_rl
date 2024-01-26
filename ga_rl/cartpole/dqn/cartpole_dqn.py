import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import glorot_uniform

from keras.optimizers import Adam
from collections import deque
import random
class DQNAgent:
    def __init__(self, state_size, action_size,batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size=batch_size
        self.replay_buffer = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration-exploitation trade-off
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        self.learning_rate = 1e-2
        self.optimizer=Adam(learning_rate=self.learning_rate)
        self.loss_function =  keras.losses.MeanSquaredError()
        self.model = self._build_model()


    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size[0], activation='relu',kernel_initializer=glorot_uniform(),bias_initializer='zeros'))
        model.add(Dense(32, activation='relu',kernel_initializer=glorot_uniform(),bias_initializer='zeros'))
        model.add(Dense(self.action_size, activation='linear',kernel_initializer=glorot_uniform(),bias_initializer='zeros'))
        model.compile(optimizer=self.optimizer,loss='mse')
        return model

    def epsilon_greedy_policy(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
   
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))


    def training_step(self):
        experiences = self.sample_experiences()
        states, actions, rewards, next_states, dones = experiences
        target =tf.reduce_max(self.model.predict(tf.reshape(next_states, (self.batch_size, self.state_size[0])), verbose=0 ),axis=1)
        target_Q_values = (rewards + (1-dones)*self.gamma*target)
        mask = tf.one_hot(actions, self.action_size)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(tf.reshape(states, (self.batch_size, self.state_size[0])))
            Q_values = tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_function(target_Q_values,Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    
    def sample_experiences(self):
        indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, done = [np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
        return states, actions, rewards, next_states, done
   
    def update_target(self):
            self.model_target.set_weights(self.model.get_weights())

def run_agent_dqn(num_episodes):

    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n    
    batch_size = 64

    agent = DQNAgent(state_size, action_size, batch_size)
    rew_table=[]

    for episode in range(num_episodes):
        state,_= env.reset()
        state = np.reshape(state, [1, state_size[0]])
        total_reward=0
        step = 0
        while True:
            action = agent.epsilon_greedy_policy(state)
            next_state, reward, done, terminated,_= env.step(action)
            next_state = np.reshape(next_state, [1, state_size[0]])

            agent.remember(state, action, reward, next_state, done)

            total_reward+=reward
            state = next_state
            if done or terminated:
                break
            step+=1
        rew_table.append(total_reward)

        if episode>50:
            agent.training_step()
        if np.mean(rew_table[episode-15:episode])>=200:
            agent.model.save_weights("./cartpole/dqn/dqn_agent.h5")
            np.save('./cartpole/dqn/rew_table_dqn.npy',np.array(rew_table))
            break

        print("Episodio {}: Reward: {}".format(episode + 1, total_reward))
        np.save('./cartpole/dqn/rew_table_dqn.npy',np.array(rew_table))
    return 

from memory_profiler import profile
@profile
def run_dqn():
    run_agent_dqn(2000)

if __name__ == "__main__":
   run_dqn()