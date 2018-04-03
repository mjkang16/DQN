import numpy as np
import tensorflow as tf
import math

TAU = 0.001
class Q_Network_D:
    """ Critic Q value model of the DDPG algorithm """
    def __init__(self, sess, num_states, output_size, seed_n, layer_size_Q1, layer_size_Q2, learning_rate_Q):
        
        self.sess = sess
        self.seed_n = seed_n
        self.learning_rate_Q = learning_rate_Q
        
        self.W1_c, self.B1_c, self.W2_c_V, self.B2_c_V, self.W2_c_A,\
        self.critic_q_model, self.critic_state_in = self.create_critic_net("", num_states, output_size, layer_size_Q1, layer_size_Q2)
        
        self.t_W1_c, self.t_B1_c, self.t_W2_c_V, self.t_B2_c_V, self.t_W2_c_A, self.t_critic_q_model, self.t_critic_state_in = self.create_critic_net("t_", num_states, output_size, layer_size_Q1, layer_size_Q2)
        
        self.saver = tf.train.Saver()
        
        self.q_value_in=tf.placeholder("float",[None,output_size])
        self.w_value_in=tf.placeholder("float",[None,output_size])
        
        self.cost = tf.reduce_mean(tf.square(self.critic_q_model-self.q_value_in))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_Q).minimize(self.cost)
        
        self.error = self.critic_q_model-self.q_value_in
        self.cost_prioritized = tf.reduce_mean(self.w_value_in*(tf.square(self.error)))
        self.optimizer_prioritized = tf.train.AdamOptimizer(learning_rate=learning_rate_Q).minimize(self.cost_prioritized)

        #initialize all tensor variable parameters:
        self.sess.run(tf.initialize_all_variables())

        #To make sure critic and target have same parmameters copy the parameters:
        # copy target parameters
        self.sess.run([
            self.t_W1_c.assign(self.W1_c),
            self.t_B1_c.assign(self.B1_c),
            self.t_W2_c_V.assign(self.W2_c_V),
            self.t_B2_c_V.assign(self.B2_c_V),
            self.t_W2_c_A.assign(self.W2_c_A)
        ])

        self.copy_target_critic_op = [
            self.t_W1_c.assign(self.W1_c),
            self.t_B1_c.assign(self.B1_c),
            self.t_W2_c_V.assign(self.W2_c_V),
            self.t_B2_c_V.assign(self.B2_c_V),
            self.t_W2_c_A.assign(self.W2_c_A)
        ]

    def create_critic_net(self, name, num_states=4, output_size=1, N_HIDDEN_1=400, N_HIDDEN_2=300):
        
        critic_state_in = tf.placeholder(tf.float32, [None, num_states], name=name+"input_x")
            
        W1 = tf.get_variable(name+"W1", shape=[num_states, N_HIDDEN_1],
                             initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
        layer1 = tf.nn.tanh(tf.matmul(critic_state_in, W1))

        W2 = tf.get_variable(name+"W2", shape=[N_HIDDEN_1, N_HIDDEN_2],
                             initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
        W3 = tf.get_variable(name+"W3", shape=[N_HIDDEN_1, N_HIDDEN_2],
                             initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))
        layer2 = tf.nn.tanh(tf.matmul(layer1, W2))
        layer3 = tf.nn.tanh(tf.matmul(layer1, W3))

        W_V = tf.get_variable(name+"W_V", shape=[N_HIDDEN_2, 1],
                             initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))

        W_A = tf.get_variable(name+"W_A", shape=[N_HIDDEN_2, output_size],
                             initializer=tf.contrib.layers.xavier_initializer(seed = self.seed_n))

        Value = tf.matmul(layer2, W_V)
        Advantage = tf.matmul(layer3, W_A)

        critic_q_model = Value + Advantage - tf.reduce_mean(Advantage, reduction_indices=1,keep_dims=True)
        
        return W1, W2, W3, W_V, W_A, critic_q_model, critic_state_in
    
    def train_critic(self, state_t_batch, y_i_batch):
        self.sess.run(self.optimizer, feed_dict={self.critic_state_in: state_t_batch, self.q_value_in: y_i_batch})
    
    def train_critic_prioritized(self, state_t_batch, y_i_batch, w_batch):
        return self.sess.run([self.error, self.cost_prioritized, self.optimizer_prioritized],
                             feed_dict={self.critic_state_in: state_t_batch, self.q_value_in: y_i_batch, self.w_value_in: w_batch})
    
    def evaluate_critic(self,state_t_1):
        return self.sess.run(self.critic_q_model, feed_dict={self.critic_state_in: state_t_1}) 
    
    def evaluate_target_critic(self,state_t_1):
        return self.sess.run(self.t_critic_q_model, feed_dict={self.t_critic_state_in: state_t_1})    
        
    def copy_target_critic(self):
        self.sess.run(self.copy_target_critic_op)

    def save_network(self, game_name, episode, save_epi):
        ep = str(episode)
        self.saver.save(self.sess, "/home/jolp/Desktop/Network/model"+ep+"_"+game_name+".ckpt")
        
    def load_network(self, game_name, saved_num):
        ep = str(saved_num)
        self.saver.restore(self.sess, "/home/jolp/Desktop/Network/model"+ep+"_"+game_name+".ckpt")
