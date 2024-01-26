import keras
import tensorflow as tf
import numpy as np
import gym
import random
import math
from keras.layers import Dense, Input
from keras.models import Model


env = gym.make("CartPole-v0")  
observation_space_shape = env.observation_space.shape
action_space_shape = env.action_space.n

def are_weights_equal(weights1,weights2):
    if(all(tf.reduce_all(tf.equal(w1, w2)) for w1, w2 in zip(weights1, weights2))):
        print("i pesi sono uguali")
    else: 
        print("i pesi sono diversi")

def sort(tau_t,fit_t ):

    sorted_indices = np.argsort(fit_t)
    t=sorted_indices[::-1]
    fit_t = fit_t[t]
    tau_t = tau_t[:, t]
    return tau_t, fit_t

class XavierInitializer(tf.keras.initializers.Initializer):
    def __init__(self, tau=None):
        self.tau=tau
        tf.random.set_seed(self.tau)
        

    def __call__(self, shape, dtype=tf.float32):

        fan_in = shape[0] if len(shape) == 1 else shape[1]
        stddev = tf.sqrt(2.0 / fan_in)
        return tf.random.normal(shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.tau)

class CPModel:
    def __init__(self, input_dim, output_dim, tau):
        self.input_dim = input_dim[0]
        self.output_dim = output_dim
        self.tau = tau
        self.initializer = XavierInitializer(self.tau)
        self.model=self.build_model()

    def build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        dense1=Dense(32, input_shape=(self.input_dim,), activation='relu',
                            kernel_initializer=self.initializer, bias_initializer='zeros', trainable=False)(input_layer)
        dense2=Dense(32, activation='relu', kernel_initializer=self.initializer, bias_initializer='zeros', trainable=False)(dense1)
        output_layer=Dense(self.output_dim, kernel_initializer=self.initializer, bias_initializer='zeros',trainable=False,activation='linear')(dense2)
        model=  Model(inputs=input_layer, outputs=output_layer)
        return model

    def get_model_weights_shapes(self):
        shapes = []
        for layer in self.model.layers:
            layer_shapes = []
            weights = layer.get_weights()
            if weights:
                layer_shapes.extend(w.shape for w in weights)
            shapes.extend(layer_shapes)
        return shapes

model_param=CPModel(observation_space_shape,action_space_shape,3)
theta_length=model_param.model.count_params()
seed_length = math.ceil(math.log2(theta_length))
model_weights_shapes=model_param.get_model_weights_shapes()
def gen_tau():
    min_value_bit = 2**seed_length
    max_value_bit = 2**(seed_length + 1)
    tau = random.randint(min_value_bit, max_value_bit - 1)
    return tau

def gen_theta0(xavier_initializer):
    theta0 = []
    for shape in model_weights_shapes:

        if len(shape) == 1:  
            theta0.append(tf.zeros(shape))
        else:
            weight = xavier_initializer(shape)
            theta0.append(weight)
    return theta0


def mutation(theta,tau,sigma):
    updated_weights = []
    np.random.seed(tau)
    shapes = [w.shape for w in theta]
    eps_matrix = [np.random.normal(0, 1, shape) for shape in shapes]
    for existing_w, new_w in zip(theta, eps_matrix):
        updated_weights.append(existing_w + sigma* new_w)  
    return updated_weights
        
def get_theta(g,n,sigma):   
    tau_table= np.load("cartpole/ga/tau_table.npy")[:g+1,n]
    theta=gen_theta0(XavierInitializer(tau_table[0]))
    if(g!=0):
        for tau in tau_table[1:g]:

            if tau!=0:
                theta=mutation(theta,tau,sigma) 


    model_param.model.set_weights(theta)


    return  model_param.model

def run_agent(g,n,sigma):
    model= get_theta(g,n,sigma)
    observation,_ = env.reset()
    sum_reward = 0
    step=0
    while True:
        state = np.reshape(observation, [1, observation_space_shape[0]])
        action_prob = model(state)
        action = np.argmax(action_prob[0])
        observation_next, reward, done, terminated ,_= env.step(action)
        observation = observation_next
        sum_reward += reward

        if done or terminated:
            observation,_ = env.reset()

            break
        step+=1

    env.close()
    return sum_reward

