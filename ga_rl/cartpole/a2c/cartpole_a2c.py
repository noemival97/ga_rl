import tensorflow as tf
import gym
import numpy as np
from keras.initializers import glorot_uniform


env = gym.make('CartPole-v0')
observation_space_shape = env.observation_space.shape[0]
action_space_shape = env.action_space.n

class A2CAgent():
    def __init__(self,observation_space_shape,action_space_shape):

        self.learning_rate_actor = 1e-3
        self.learning_rate_critic = 1e-2
        self.gamma = 0.99
        self.observation_space_shape=observation_space_shape
        self.action_space_shape=action_space_shape
        self.actor_optimizer=tf.keras.optimizers.Adam(learning_rate= self.learning_rate_actor)
        self.critic_optimizer=tf.keras.optimizers.Adam(learning_rate= self.learning_rate_critic)

        self.actor=self.build_actor()
        self.critic=self.build_critic()

    def build_actor(self):
        actor= tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu', input_shape=(self.observation_space_shape,), kernel_initializer=glorot_uniform(),bias_initializer='zeros'),
                tf.keras.layers.Dense(self.action_space_shape, activation='softmax')
                ]) 
        return actor
    
    def build_critic(self):
        critic= tf.keras.Sequential([
                    tf.keras.layers.Dense(32, activation='relu', input_shape=(self.observation_space_shape,),kernel_initializer=glorot_uniform(),bias_initializer='zeros'),
                    tf.keras.layers.Dense(1)
                    ])

        return critic
    def take_action(self,state):
        policy=self.actor.predict(state,verbose=0)[0]
        return np.random.choice(range(len(policy)), p=policy)

    def update_weights(self,state, action, reward, next_state, done):
            
        with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
            target = reward + (1 - done) * self.gamma * self.critic(next_state)
            critic_value = self.critic(state)

            advantage = target - critic_value

            action_probabilities = self.actor(state)
            selected_action_probability = action_probabilities[0, action]
            actor_loss = -tf.math.log(selected_action_probability) * advantage

            critic_loss = tf.keras.losses.mean_squared_error(critic_value,target)

        gradients_actor = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
        gradients_critic = tape_critic.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(gradients_actor, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(gradients_critic, self.critic.trainable_variables))
        
def run_agent_ac(num_episodes):
    rew_table=[]
    actor_critic=A2CAgent(observation_space_shape,action_space_shape)

    for episode in range(num_episodes):
        state,_ = env.reset()
        state = np.expand_dims(state, axis=0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        total_reward = 0
        step = 0
        while True:
            action = actor_critic.take_action(state)
            next_state, reward, done, terminated ,_= env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
            actor_critic.update_weights(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done or terminated:
                break
            step+=1
        rew_table.append(total_reward)

        if np.mean(rew_table[episode-15:episode])>=200:
            actor_critic.actor.save_weights("./cartpole/a2c/actor_agent.h5")
            actor_critic.critic.save_weights('/cartpole/a2c/critic.h5')
            np.save('./cartpole/a2c/rew_table_ac.npy',np.array(rew_table))
            break

        print("Episodio {}: Reward: {}".format(episode + 1, total_reward))
        np.save('./cartpole/a2c/rew_table_ac.npy',np.array(rew_table))

    return 


from memory_profiler import profile

@profile
def run_ac():

    run_agent_ac(2000)


if __name__ == "__main__":
   run_ac()