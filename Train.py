

import numpy as np
import tensorflow as tf
import Exploration
import time

GAMMA = 0.99

def train_prioritized(Q_Network, train_batch, w_batch, Exp, s_scale, input_size, num_actions):
    
    state_t_batch = [item[0] for item in train_batch]
    state_t_batch = np.array(state_t_batch)
    state_t_1_batch = [item[1] for item in train_batch]
    state_t_1_batch = np.array(state_t_1_batch)
    
    action_batch = [item[2] for item in train_batch]
    reward_batch = [item[3] for item in train_batch]
    reward_batch = np.array(reward_batch)
    done_batch = [item[4] for item in train_batch]
    done_batch = np.array(done_batch)
    
    action_t_batch = Q_Network.evaluate_critic(state_t_batch)
    w_batch = np.transpose(np.tile(w_batch, (num_actions, 1)))
    
    if Exp == 'epsilon':
        action_t_1_batch = Q_Network.evaluate_critic(state_t_1_batch)
        q_t_1 = Q_Network.evaluate_target_critic(state_t_1_batch)
        
        for i in range(0, len(train_batch)):
            if done_batch[i]:
                action_t_batch[i][action_batch[i]] = reward_batch[i]
            else:
                action_t_batch[i][action_batch[i]] = reward_batch[i] + GAMMA*q_t_1[i][np.argmax(action_t_1_batch[i])]
    
    elif Exp == 'softmax':
        action_t_1_batch = Q_Network.evaluate_target_critic(state_t_1_batch)
        q_t_1 = Exploration.softV(action_t_1_batch, s_scale)
        
        for i in range(0, len(train_batch)):
            if done_batch[i]:
                action_t_batch[i][action_batch[i]] = reward_batch[i]
            else:
                action_t_batch[i][action_batch[i]] = reward_batch[i] + GAMMA*q_t_1[i]
    
    elif Exp == 'sparsemax':
        action_t_1_batch = Q_Network.evaluate_target_critic(state_t_1_batch)
        q_t_1 = Exploration.sparsemax(action_t_1_batch, s_scale)
        
        for i in range(0, len(train_batch)):
            if done_batch[i]:
                action_t_batch[i][action_batch[i]] = reward_batch[i]
            else:
                action_t_batch[i][action_batch[i]] = reward_batch[i] + GAMMA*q_t_1[i]
    
    #Q_Network.train_critic_prioritized(state_t_batch, action_t_batch, w_batch)
    errors, cost, _ = Q_Network.train_critic_prioritized(state_t_batch, action_t_batch, w_batch)
    errors = np.sum(errors, axis=1)
    return errors, cost
"""
errors = []
for i in range(batch_size):
    state_m, next_state_m, action_m, reward_m, done_m = minibatch[i]
    q_t = np.max(Q_Network.evaluate_critic(np.reshape(state_m, [1, input_size])))
    q_t_1 = np.max(Q_Network.evaluate_critic(np.reshape(next_state_m, [1, input_size])))
    if done_m:
        q_t_1 = reward_m
    else:
        q_t_1 = reward_m + dis*q_t_1
    errors.append(q_t_1-q_t)
"""
def train_error(Q_Network, train_batch, Exp, s_scale, input_size, num_actions):
    
    state_t_batch = [item[0] for item in train_batch]
    state_t_batch = np.array(state_t_batch)
    state_t_1_batch = [item[1] for item in train_batch]
    state_t_1_batch = np.array(state_t_1_batch)
    
    action_batch = [item[2] for item in train_batch]
    reward_batch = [item[3] for item in train_batch]
    reward_batch = np.array(reward_batch)
    done_batch = [1 if item[4]==False else 0 for item in train_batch]
    done_batch = np.array(done_batch)
    
    q_t_batch = Q_Network.evaluate_critic(state_t_batch)
    q_t = [q_t_batch[i][action_batch[i]] for i in range(len(train_batch))]
    q_t = np.array(q_t)
    
    action_t_1_batch = Q_Network.evaluate_critic(state_t_1_batch)
    q_t_1_batch = Q_Network.evaluate_target_critic(state_t_1_batch)
    
    if Exp == 'epsilon':
        q_t_1 = [q_t_1_batch[i][np.argmax(action_t_1_batch[i])] for i in range(len(train_batch))]
        q_t_1 = np.array(q_t_1)
        error_batch = GAMMA*q_t_1*done_batch + reward_batch - q_t
    
    elif Exp == 'softmax':
        q_t_1 = Exploration.softV(q_t_1_batch, s_scale)
        error_batch = GAMMA*q_t_1*done_batch + reward_batch - q_t
    
    elif Exp == 'sparsemax':
        #q_t_1 = Exploration.sparsemax(q_t_1_batch, s_scale)
        #error_batch = GAMMA*q_t_1*done_batch + reward_batch - q_t
        q_t_1 = [q_t_1_batch[i][np.argmax(action_t_1_batch[i])] for i in range(len(train_batch))]
        q_t_1 = np.array(q_t_1)
        error_batch = GAMMA*q_t_1*done_batch + reward_batch - q_t
    
    return error_batch
    
def train(Q_Network, train_batch, Exp, s_scale, input_size, num_actions):
    
    state_t_batch = [item[0] for item in train_batch]
    state_t_batch = np.array(state_t_batch)
    state_t_1_batch = [item[1] for item in train_batch]
    state_t_1_batch = np.array(state_t_1_batch)
    
    action_batch = [item[2] for item in train_batch]
    reward_batch = [item[3] for item in train_batch]
    reward_batch = np.array(reward_batch)
    done_batch = [item[4] for item in train_batch]
    done_batch = np.array(done_batch)
    
    action_t_batch = Q_Network.evaluate_critic(state_t_batch)
    
    if Exp == 'epsilon':
        action_t_1_batch = Q_Network.evaluate_critic(state_t_1_batch)
        q_t_1 = Q_Network.evaluate_target_critic(state_t_1_batch)
        
        for i in range(0, len(train_batch)):
            if done_batch[i]:
                action_t_batch[i][action_batch[i]] = reward_batch[i]
            else:
                action_t_batch[i][action_batch[i]] = reward_batch[i] + GAMMA*q_t_1[i][np.argmax(action_t_1_batch[i])]
    
    elif Exp == 'softmax':
        action_t_1_batch = Q_Network.evaluate_target_critic(state_t_1_batch)
        q_t_1 = Exploration.softV(action_t_1_batch, s_scale)
        
        for i in range(0, len(train_batch)):
            if done_batch[i]:
                action_t_batch[i][action_batch[i]] = reward_batch[i]
            else:
                action_t_batch[i][action_batch[i]] = reward_batch[i] + GAMMA*q_t_1[i]
    
    elif Exp == 'sparsemax':
        action_t_1_batch = Q_Network.evaluate_target_critic(state_t_1_batch)
        q_t_1 = Exploration.sparsemax(action_t_1_batch, s_scale)
        
        for i in range(0, len(train_batch)):
            if done_batch[i]:
                action_t_batch[i][action_batch[i]] = reward_batch[i]
            else:
                action_t_batch[i][action_batch[i]] = reward_batch[i] + GAMMA*q_t_1[i]
    
    # Update critic by minimizing the loss
    Q_Network.train_critic(state_t_batch, action_t_batch)

def copy(Q_Network):
    # Update target Critic
    Q_Network.copy_target_critic()

    
def if_prioritized(Q_Network, train_batch, input_size, num_actions, eps, alpha):
    
    state_t_batch = [item[0] for item in train_batch]
    state_t_batch = np.array(state_t_batch)
    state_t_1_batch = [item[1] for item in train_batch]
    state_t_1_batch = np.array(state_t_1_batch)
    
    action_batch = [item[2] for item in train_batch]
    action_batch = np.array(action_batch)
    action_batch = np.reshape(action_batch, [len(action_batch), num_actions])
    reward_batch = [item[3] for item in train_batch]
    reward_batch = np.array(reward_batch)
    done_batch = [item[4] for item in train_batch]
    done_batch = np.array(done_batch)
    
    q_t_1 = Q_Network.evaluate_critic(state_t_1_batch)
    
    y_i_batch=[]
    for i in range(0, len(train_batch)):
        if done_batch[i]:
            y_i_batch.append(reward_batch[i])
        else:
            y_i_batch.append(reward_batch[i] + GAMMA*np.max(q_t_1[i]))
    
    y_i_batch = np.array(y_i_batch)
    y_i_batch = np.reshape(y_i_batch,[len(y_i_batch),1])
    
    action_t_batch = Q_Network.evaluate_critic(state_t_batch)
    
    TD_i_batch = []
    for i in range(0, len(train_batch)):
        TD_i_batch.append(y_i_batch[i][0]-np.max(action_t_batch[i]))
    TD_i_batch = [np.power(abs(TD_i_batch[i])+eps,alpha) for i in range(len(train_batch))]
    
    sum_TD = np.sum(TD_i_batch)
    TD_i_batch = np.array(TD_i_batch) / float(sum_TD)
    
    return TD_i_batch
    
    
    
    
