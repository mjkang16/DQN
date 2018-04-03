import numpy as np
import tensorflow as tf
import math

TAU = 0.001
class Q_Network:
    """ Critic Q value model of the DDPG algorithm """
    def __init__(self, sess, num_states, output_size, seed_n, layer_size_Q1, layer_size_Q2, learning_rate_Q):
        
        self.sess = sess
        self.seed_n = seed_n
        
        #critic_q_model parameters:
        self.W1_c, self.B1_c, self.W2_c_A, self.B2_c_A, self.W3_c_A, self.B3_c_A,\
        self.critic_q_model, self.critic_state_in = self.create_critic_net("", num_states, output_size, layer_size_Q1, layer_size_Q2)
        
        #create target_q_model:
        self.t_W1_c, self.t_B1_c, self.t_W2_c_A, self.t_B2_c_A, self.t_W3_c_A, self.t_B3_c_A,\
        self.t_critic_q_model, self.t_critic_state_in = self.create_critic_net("t_", num_states, output_size, layer_size_Q1, layer_size_Q2)

        self.q_value_in=tf.placeholder("float",[None,output_size])
        
        self.cost = tf.reduce_mean(tf.square(self.critic_q_model-self.q_value_in))
        #self.cost=tf.pow(self.critic_q_model-self.q_value_in,2)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_Q).minimize(self.cost)

        #initialize all tensor variable parameters:
        self.sess.run(tf.initialize_all_variables())

        #To make sure critic and target have same parmameters copy the parameters:
        # copy target parameters
        self.sess.run([
            self.t_W1_c.assign(self.W1_c),
            self.t_B1_c.assign(self.B1_c),
            self.t_W2_c_A.assign(self.W2_c_A),
            self.t_B2_c_A.assign(self.B2_c_A),
            self.t_W3_c_A.assign(self.W3_c_A),
            self.t_B3_c_A.assign(self.B3_c_A)
        ])

        self.copy_target_critic_op = [
            self.t_W1_c.assign(self.W1_c),
            self.t_B1_c.assign(self.B1_c),
            self.t_W2_c_A.assign(self.W2_c_A),
            self.t_B2_c_A.assign(self.B2_c_A),
            self.t_W3_c_A.assign(self.W3_c_A),
            self.t_B3_c_A.assign(self.B3_c_A)
        ]

    def create_critic_net(self, name, num_states=4, output_size=1, N_HIDDEN_1=400, N_HIDDEN_2=300):
        
        critic_state_in = tf.placeholder("float",[None,num_states])  
    
        W1_c = tf.get_variable(name+"W1_c", shape=[num_states,N_HIDDEN_1],\
                                 initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
        B1_c = tf.get_variable(name+"B1_c", shape=[N_HIDDEN_1],\
                                 initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
        
        W2_c_A = tf.get_variable(name+"W2_c_A", shape=[N_HIDDEN_1,N_HIDDEN_2],\
                                 initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
        B2_c_A = tf.get_variable(name+"B2_c_A", shape=[N_HIDDEN_2],\
                                 initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
        
        W3_c_A = tf.get_variable(name+"W3_c_A", shape=[N_HIDDEN_2,output_size],\
                                 initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
        B3_c_A = tf.get_variable(name+"B3_c_A", shape=[output_size],\
                                 initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
        
        H1_c=tf.nn.tanh(tf.matmul(critic_state_in,W1_c)+B1_c)
        
        H2_c_A=tf.nn.tanh(tf.matmul(H1_c,W2_c_A)+B2_c_A)
        
        critic_q_model= tf.matmul(H2_c_A,W3_c_A)+B3_c_A
        
        return W1_c, B1_c, W2_c_A, B2_c_A, W3_c_A, B3_c_A, critic_q_model, critic_state_in
    
    def train_critic(self, state_t_batch, y_i_batch ):
        self.sess.run(self.optimizer, feed_dict={self.critic_state_in: state_t_batch, self.q_value_in: y_i_batch})
    
    def evaluate_critic(self,state_t_1):
        return self.sess.run(self.critic_q_model, feed_dict={self.critic_state_in: state_t_1}) 
    
    def evaluate_target_critic(self,state_t_1):
        return self.sess.run(self.t_critic_q_model, feed_dict={self.t_critic_state_in: state_t_1})    
        
    def copy_target_critic(self):
        self.sess.run(self.copy_target_critic_op)
