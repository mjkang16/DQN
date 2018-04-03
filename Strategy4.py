
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import random
from collections import deque
import gym
from Q_Network import Q_Network
from Q_Network_Dueling import Q_Network_D
import Train
import Exploration
import replaymemory
import matplotlib.pyplot as plt
import pickle
import time

class RL:
    def __init__(self, sess, dis = 0.99, REPLAY_MEMORY = 10000, batch_size = 64, max_episodes = 500,
                 
                 layer_size_Q1 = 64, layer_size_Q2 = 64, learning_rate_Q = 0.0001,
                 training_step = 1, copy_step = 1, repu_num = 1, action_res = None,
                 
                 ending_cond_epis = 100, ending_cond_reward = 195,
                 alpha = 0.6, beta_init = 0.4, eps = 0.01, eps_div = 256, s_scale = 1,
                 
                 Double = True, Dueling = True, Prioritized = True, Exp = 'softmax',
                 seed_n = 0, Game = 'CartPole-v0', file_name = 'steps', case_n = 0, save_epi = 100):
        
        env = gym.make(Game)
        tf.set_random_seed(seed_n)
        rng = np.random.RandomState(seed_n)
        
        input_size = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        #self.action_dim = 1
        output_size, conti_action_flag, action_map = self.get_action_information(env, Game, action_res=action_res)
        
        replay_memory = replaymemory.ReplayMemory(rng=rng, memory_size=REPLAY_MEMORY, per_alpha=alpha, per_beta0=beta_init)
        
        self.sess = sess
        self.dis = dis
        self.REPLAY_MEMORY = REPLAY_MEMORY
        self.replay_memory_size = REPLAY_MEMORY
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        
        self.alpha = alpha
        self.beta_init = beta_init
        self.eps = eps
        self.eps_div = eps_div
        self.s_scale = s_scale
        
        self.layer_size_Q1 = layer_size_Q1
        self.layer_size_Q2 = layer_size_Q2
        self.learning_rate_Q = learning_rate_Q
        
        self.training_step = training_step
        self.copy_step = copy_step
        self.repu_num = repu_num
        
        self.Double = Double
        self.Dueling = Dueling
        self.Prioritized = Prioritized
        self.Exp = Exp
        
        self.seed_n = seed_n
        self.Game = Game
        self.save_epi = save_epi
        self.max_episodes = max_episodes
        self.env = env
        self.file_name = file_name
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.conti_action_flag=conti_action_flag
        self.action_map=action_map
        
        if Dueling:
            self.Q_Network = Q_Network_D(sess, input_size, output_size, seed_n, layer_size_Q1, layer_size_Q2, learning_rate_Q)
        else:
            self.Q_Network = Q_Network(sess, input_size, output_size, seed_n, layer_size_Q1, layer_size_Q2, learning_rate_Q)
        
        self.ending_cond_epis = ending_cond_epis
        self.ending_cond_reward = ending_cond_reward
        
        self.run_DQN(case_n = case_n, seed_n = seed_n, Exp = Exp, Double = Double, Dueling = Dueling, Prioritized = Prioritized)

        
    def get_action_information(self, env, env_name, action_res=None):
        action_map = []
        if isinstance(env.action_space, gym.spaces.Box):
            conti_action_flag = True
            if env_name == "Pendulum-v0" or env_name == "InvertedPendulum-v1" or env_name == "MountainCarContinuous-v0" or env_name == "InvertedDoublePendulum-v1":
                action_map = np.linspace(env.action_space.low[0],env.action_space.high[0],num=action_res)
            
            elif env_name == "Reacher-v1":
                action_map = np.zeros([np.prod(action_res), 2])
                u = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                v = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                for i in range(action_res[0]):
                    for j in range(action_res[1]):
                        s = action_res[1] * i + j
                        action_map[s, :] = [u[i], v[j]]
            
            elif env_name == "Swimmer-v1" or env_name == "LunarLanderContinuous-v2" or env_name == "MultiGoal-v0":
                action_map = np.zeros([np.prod(action_res), 2])
                u = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                v = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                for i in range(action_res[0]):
                    for j in range(action_res[1]):
                        s = action_res[1] * i + j
                        action_map[s, :] = [u[i], v[j]]
            
            elif env_name == "Hopper-v1":
                action_map = np.zeros([np.prod(action_res), 3])
                u = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                v = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                w = np.linspace(env.action_space.low[2], env.action_space.high[2], num=action_res[2])
                for i in range(action_res[0]):
                    for j in range(action_res[1]):
                        for k in range(action_res[2]):
                            s = action_res[2] * action_res[1] * i + action_res[2] * j + k
                            action_map[s, :] = [u[i], v[j], w[k]]
            
            elif env_name == "Walker2d-v1" or env_name == "HalfCheetah-v1":
                action_map = np.zeros([np.prod(action_res), 6])
                x = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                y = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                z = np.linspace(env.action_space.low[2], env.action_space.high[2], num=action_res[2])
                u = np.linspace(env.action_space.low[3], env.action_space.high[3], num=action_res[3])
                v = np.linspace(env.action_space.low[4], env.action_space.high[4], num=action_res[4])
                w = np.linspace(env.action_space.low[5], env.action_space.high[5], num=action_res[5])
                for i0 in range(action_res[0]):
                    for i1 in range(action_res[1]):
                        for i2 in range(action_res[2]):
                            for i3 in range(action_res[3]):
                                for i4 in range(action_res[4]):
                                    for i5 in range(action_res[5]):
                                        s = np.prod(action_res[1:]) * i0
                                        s += np.prod(action_res[2:]) * i1
                                        s += np.prod(action_res[3:]) * i2
                                        s += np.prod(action_res[4:]) * i3
                                        s += np.prod(action_res[5:]) * i4
                                        s += i5
                                        action_map[s, :] = [x[i0], y[i1], z[i2], u[i3], v[i4], w[i5]]
            elif env_name == "Ant-v1":
                action_map = np.zeros([np.prod(action_res), 8])
                x = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])
                y = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])
                z = np.linspace(env.action_space.low[2], env.action_space.high[2], num=action_res[2])
                u = np.linspace(env.action_space.low[3], env.action_space.high[3], num=action_res[3])
                v = np.linspace(env.action_space.low[4], env.action_space.high[4], num=action_res[4])
                w = np.linspace(env.action_space.low[5], env.action_space.high[5], num=action_res[5])
                p = np.linspace(env.action_space.low[6], env.action_space.high[6], num=action_res[6])
                q = np.linspace(env.action_space.low[7], env.action_space.high[7], num=action_res[7])
                for i0 in range(action_res[0]):
                    for i1 in range(action_res[1]):
                        for i2 in range(action_res[2]):
                            for i3 in range(action_res[3]):
                                for i4 in range(action_res[4]):
                                    for i5 in range(action_res[5]):
                                        for i6 in range(action_res[6]):
                                            for i7 in range(action_res[7]):
                                                s = np.prod(action_res[1:]) * i0
                                                s += np.prod(action_res[2:]) * i1
                                                s += np.prod(action_res[3:]) * i2
                                                s += np.prod(action_res[4:]) * i3
                                                s += np.prod(action_res[5:]) * i4
                                                s += np.prod(action_res[6:]) * i5
                                                s += np.prod(action_res[7:]) * i6
                                                s += i7
                                        action_map[s, :] = [x[i0], y[i1], z[i2], u[i3], v[i4], w[i5], p[i6], q[i7]]
            else:
                print(env.action_space.high.shape[0])
            n_action = np.prod(action_res)
        
        elif isinstance(env.action_space, gym.spaces.Discrete):
            conti_action_flag = False
            n_action = env.action_space.n
        
        else:
            raise NotImplementedError("{} action spaces are not supported yet.".format(type(env.action_space)))
        return n_action, conti_action_flag, action_map
    
    def format_experience(self, experience, minibatch):
        states_b, actions_b, rewards_b, states_n_b, done_b = zip(*experience)
        
        for num in range(len(states_b)):
            minibatch.append((states_b[num],states_n_b[num],actions_b[num],rewards_b[num],done_b[num]))
        return minibatch
        
    def run_DQN(self, case_n, seed_n, Exp, Double, Dueling, Prioritized):
        sess = self.sess
        dis = self.dis
        REPLAY_MEMORY = self.REPLAY_MEMORY
        replay_memory = self.replay_memory
        batch_size = self.batch_size

        Game = self.Game
        save_epi = self.save_epi
        max_episodes = self.max_episodes
        env = self.env
        input_size = self.input_size
        output_size = self.output_size
        
        alpha = self.alpha
        beta_init = self.beta_init
        eps = self.eps
        eps_div = self.eps_div
        s_scale = self.s_scale
        
        training_step = self.training_step
        copy_step = self.copy_step
        repu_num = self.repu_num
        
        ending_cond_epis = self.ending_cond_epis
        ending_cond_reward = self.ending_cond_reward
        
        conti_action_flag = self.conti_action_flag
        action_map = self.action_map
        
        env.seed(seed_n)
        np.random.seed(seed_n)
        tf.set_random_seed(seed_n)
        random.seed(seed_n)

        Q_Network = self.Q_Network
        
        end_episode = 0
        step_count_total = 0
        global_step = 0
        loss = 0

        replay_buffer = deque()
        Q_list = []
        TD_buffer = deque()
        steps_list = []
        step_avg_list = []
        global_step_list = []

        print("")
        print("CASE {}".format(case_n))
        print("  STATE DIM : {}, ACTION DIM : {}".format(input_size, self.action_dim))
        print("  Exp : {}".format(Exp))
        print("  Strategy : Double : {}, Dueling : {}, Prioritized : {}".format(Double, Dueling, Prioritized))
        
        t = t1 = t2 = t3 = t4 = t5 = t6 = t7 = t8 = 0
        for episode in range(1, max_episodes+1):
            t1 = time.time()
            #print(    "TIME {} --- EPISODE {}".format(t1-t,episode))
            t = t1
            
            done = False
            step_count = 0
            current_step = 0
            cost = 0
            state = env.reset()
            
            while not done:
                t7 = time.time()
                e = 1. / ((float(episode - 1) / eps_div) + 1)
                
                action = Exploration.choice_action(Exp, e, s_scale, Q_Network.evaluate_critic(np.reshape(state, [1, input_size]))[0])
                
                if conti_action_flag:
                    action0 = [action_map[action]]
                else:
                    action0 = action
                    
                next_state, reward, done, _ = env.step(action0)
                step_count += reward
                global_step += 1
                current_step += 1
                
                
                #t2 = time.time()
                #print(    "TIME {} --- 1 {}  {}".format(t2-t7,episode,current_step))
                if Prioritized:
                    replay_memory.save_experience(state, action, reward, next_state, done)
                    
                else:
                    replay_buffer.append((state, next_state, action, reward, done))
                    if len(replay_buffer) > REPLAY_MEMORY:
                        replay_buffer.popleft()
                #t2 = time.time()
                #print(    "TIME {} --- 2 {}  {}".format(t2-t7,episode,current_step))
                
                state = next_state
                
                replay_memory.anneal_per_importance_sampling(global_step, max_episodes*1000)
                if global_step > batch_size and global_step % training_step == 0:
                    for re in range(repu_num):
                        minibatch = []
                        if Prioritized:
                            idx, priorities, w_batch, experience = replay_memory.retrieve_experience(batch_size)
                            minibatch = self.format_experience(experience, minibatch)
                            errors, cost = Train.train_prioritized(Q_Network, minibatch, w_batch, Exp, s_scale, input_size, output_size)
                            #print(errors)
                            
                            #t2 = time.time()
                            #print(    "TIME {} --- 3 {}  {}".format(t2-t7,episode,current_step))
                            
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
                            #errors = Train.train_error(Q_Network, minibatch, Exp, s_scale, input_size, output_size)
                            #t2 = time.time()
                            #print(    "TIME {} --- 4 {}  {}".format(t2-t7,episode,current_step))   
                            replay_memory.update_experience_weight(idx, errors)
                          
                        else:
                            minibatch = random.sample(replay_buffer, batch_size)
                            Train.train(Q_Network, minibatch, Exp, s_scale, input_size, self.action_dim)
                        
                
                if global_step > batch_size and global_step % copy_step == 0:
                    Train.copy(Q_Network)
                
                #t8 = time.time()
                #print(    "TIME {} --- CYCLE {}  {}".format(t8-t7,episode,current_step)) 
                
            steps_list.append(step_count)
            global_step_list.append(global_step)
            
            # Print the average of result 
            if episode < ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_avg_list.append(step_count_total / episode)

            if episode == ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_avg_list.append(step_count_total / ending_cond_epis)

            if episode > ending_cond_epis:
                step_count_total += steps_list[episode - 1]
                step_count_total -= steps_list[episode - 1 - ending_cond_epis]
                step_avg_list.append(step_count_total / ending_cond_epis)
            
            print("{}           {}".format(episode, round(step_avg_list[episode - 1], 3)))
            if Exp == 'epsilon' or Exp == 'sparsemax':
                print ("                   ( Result : {},  Loss : {},  Epsilon : {},  Steps : {},  Global Steps : {} )"
                                   .format(round(step_count, 5), round(cost, 5), round(e, 5), current_step, global_step))
            else:
                print ("                   ( Result : {},  Loss : {},  Steps : {},  Global Steps : {} )"
                                   .format(round(step_count, 5), round(cost, 5), current_step, global_step))
            
            # Save the networks 
            if episode % save_epi == 0:
                file_case = str(case_n)
                Q_Network.save_network(game_name = self.file_name+'_seed'+file_case, episode = episode, save_epi = save_epi)
            
                with open('/home/jolp/Desktop/Data/'+self.file_name+'_seed'+file_case, 'wb') as fout:
                    pickle.dump(step_avg_list, fout)
                with open('/home/jolp/Desktop/Data/'+self.file_name+'_global_'+'_seed'+file_case, 'wb') as fout2:
                    pickle.dump(global_step_list, fout2)

                x_values = list(range(1, episode+1))
                y_values = step_avg_list[:]
                plt.plot(x_values, y_values, c='green')
                plt.title(self.file_name)
                plt.grid(True)
                plt.show()
            
            end_episode += 1
            
            if step_avg_list[episode - 1] > ending_cond_reward:
                break

        print("--------------------------------------------------")
        print("--------------------------------------------------")
        for episode in range(end_episode + 1, max_episodes+1):
            s = env.reset()
            reward_sum = 0
            done = False
            while not done :
                #env.render()
                action = np.argmax(Q_Network.evaluate_critic(np.reshape(state, [1, input_size])))

                if conti_action_flag:
                    action = [action_map[action]]
                else:
                    action = action
                
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                global_step += 1

                #if episode % save_epi == 0:
                #    Q_Network.save_network(episode = episode, save_epi = save_epi)
                #    Action_Network.save_network(episode = episode, save_epi = save_epi)

                if done :
                    steps_list.append(reward_sum)
                    global_step_list.append(global_step)
                    step_count_total += steps_list[episode - 1]
                    step_count_total -= steps_list[episode - 1 - ending_cond_epis]
                    step_avg_list.append(step_count_total / ending_cond_epis)
                    print("{}           {}".format(episode, round(step_avg_list[episode - 1], 3)))
                    print ("                   ( Result : {} )".format(reward_sum))
        
            if episode % save_epi == 0:
                file_case = str(case_n)
                Q_Network.save_network(game_name = self.file_name+'_seed'+file_case, episode = episode, save_epi = save_epi)
                with open('/home/jolp/Desktop/Data/'+self.file_name+'_seed'+file_case, 'wb') as fout:
                    pickle.dump(step_avg_list, fout)
                with open('/home/jolp/Desktop/Data/'+self.file_name+'_global_'+'_seed'+file_case, 'wb') as fout2:
                    pickle.dump(global_step_list, fout2)

                x_values = list(range(1, episode+1))
                y_values = step_avg_list[:]
                plt.plot(x_values, y_values, c='green')
                plt.title(self.file_name)
                plt.grid(True)
                plt.show()
        
        
        file_case = str(case_n)
        with open('/home/jolp/Desktop/Data/'+self.file_name+'_seed'+file_case, 'wb') as fout:
            pickle.dump(step_avg_list, fout)
        with open('/home/jolp/Desktop/Data/'+self.file_name+'_global_'+'_seed'+file_case, 'wb') as fout2:
            pickle.dump(global_step_list, fout2)
        
        x_values = list(range(1, max_episodes+1))
        y_values = step_avg_list[:]
        plt.plot(x_values, y_values, c='green')
        plt.title(self.file_name)
        plt.grid(True)
        plt.show()
        
