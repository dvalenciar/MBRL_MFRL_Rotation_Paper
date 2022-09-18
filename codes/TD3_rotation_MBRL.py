
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from Memory_MBRL  import MemoryClassMB
from main_rl_env_rotation_v2 import RL_ENV
from Networks import Actor, Critic, ModelNet_probabilistic_transition


class TD3agent_rotation_mbrl:

    def __init__(self, env):

        # -------- Hyper-parameters --------------- #
        self.env   = env
        self.gamma = 0.99  # discount factor
        self.tau   = 0.005

        self.batch_size_policy = 32
        self.batch_size_model  = 32

        self.G = 5
        self.M = 20
        self.R = 1

        self.update_counter     = 0
        self.policy_freq_update = 2

        self.max_memory_size_env   = 20_000
        self.max_memory_size_model = 40_000

        self.actor_learning_rate      = 1e-4
        self.critic_learning_rate     = 1e-3
        self.transition_learning_rate = 0.001

        self.hidden_size_critic        = [128, 64, 32]
        self.hidden_size_actor         = [128, 64, 32]

        self.hidden_size_network_model = [32, 32, 32]

        # -------- Parameters --------------- #
        self.num_states  = 16
        self.num_actions = 4
        self.num_states_training = 15  # 16 in total but remove the goal angle from the state for training trans model

        # how often to choose the "imagine data"
        self.epsilon       = 1
        self.epsilon_min   = 0.001
        self.epsilon_decay = 0.0001

        # ------------- Initialization memory --------------------- #
        self.memory = MemoryClassMB(self.max_memory_size_env, self.max_memory_size_model)

        # ---------- Initialization and build the networks ----------- #
        # Main networks
        self.actor     = Actor(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_q1 = Critic(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)
        self.critic_q2 = Critic(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Target networks
        self.actor_target     = Actor(self.num_states, self.hidden_size_actor, self.num_actions)
        self.critic_target_q1 = Critic(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)
        self.critic_target_q2 = Critic(self.num_states + self.num_actions, self.hidden_size_critic, self.num_actions)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer    = optim.Adam(self.actor.parameters(),     lr=self.actor_learning_rate)
        self.critic_optimizer_1 = optim.Adam(self.critic_q1.parameters(), lr=self.critic_learning_rate)
        self.critic_optimizer_2 = optim.Adam(self.critic_q2.parameters(), lr=self.critic_learning_rate)

        # ---------- Initialization and build the networks for Model Learning ----------- #
        self.pdf_transition_model_1 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)
        self.pdf_transition_model_2 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)
        self.pdf_transition_model_3 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)
        self.pdf_transition_model_4 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)
        self.pdf_transition_model_5 = ModelNet_probabilistic_transition(self.num_states_training + self.num_actions,
                                                                        self.hidden_size_network_model)

        self.pdf_transition_1_optimizer = optim.Adam(self.pdf_transition_model_1.parameters(),
                                                     lr=self.transition_learning_rate)
        self.pdf_transition_2_optimizer = optim.Adam(self.pdf_transition_model_2.parameters(),
                                                     lr=self.transition_learning_rate)
        self.pdf_transition_3_optimizer = optim.Adam(self.pdf_transition_model_3.parameters(),
                                                     lr=self.transition_learning_rate)
        self.pdf_transition_4_optimizer = optim.Adam(self.pdf_transition_model_4.parameters(),
                                                     lr=self.transition_learning_rate)
        self.pdf_transition_5_optimizer = optim.Adam(self.pdf_transition_model_5.parameters(),
                                                     lr=self.transition_learning_rate)

    def get_action_from_policy(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # numpy to a tensor with shape [1,15]
        with torch.no_grad():
            self.actor.eval()
            action = self.actor.forward(state_tensor)
            action = action.detach()
            action = action.numpy()  # tensor to numpy
            self.actor.train()
        return action[0]

    def add_real_experience_memory(self, state, action, reward, next_state, done):
        self.memory.replay_buffer_environment_add(state, action, reward, next_state, done)

    def add_imagined_experience_memory(self, state, action, reward, next_state, done):
        self.memory.replay_buffer_model_add(state, action, reward, next_state, done)

    def epsilon_greedy_function_update(self):
        # this is used for choose the sample from memory when dream samples are generated
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)

    def step_training(self, data_type):
        # check, if enough samples are available in memory
        if self.memory.len_env_buffer() and self.memory.len_model_buffer() <= self.batch_size_policy:
            return
        else:
            self.update_weights(data_type)

    def update_weights(self, data_type):

        for it in range(1, self.G + 1):

            self.update_counter += 1

            if data_type == 'env_data':
                states, actions, rewards, next_states, dones = self.memory.sample_experience_from_env(self.batch_size_policy)

            elif data_type == 'model_data':
                states, actions, rewards, next_states, dones = self.memory.sample_experience_from_model(self.batch_size_policy)

            else:
                print("no data to sample")
                return

            states  = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            dones   = np.array(dones).reshape(-1, 1)
            next_states = np.array(next_states)

            states  = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones   = torch.FloatTensor(dones)
            next_states = torch.FloatTensor(next_states)

            # ------- compute the target action
            next_actions = self.actor_target.forward(next_states)

            # add noise also here, paper mention this
            next_actions = next_actions.detach().numpy()  # tensor to numpy
            next_actions = next_actions + (np.random.normal(0, scale=0.2, size=self.num_actions))
            next_actions = np.clip(next_actions, -1, 1)
            next_actions = torch.FloatTensor(next_actions)

            # compute next targets values
            next_Q_vales_q1 = self.critic_target_q1.forward(next_states, next_actions)
            next_Q_vales_q2 = self.critic_target_q2.forward(next_states, next_actions)

            q_min = torch.minimum(next_Q_vales_q1, next_Q_vales_q2)

            Q_target = rewards + (self.gamma * (1 - dones) * q_min).detach()

            loss = nn.MSELoss()

            Q_vals_q1 = self.critic_q1.forward(states, actions)
            Q_vals_q2 = self.critic_q2.forward(states, actions)

            critic_loss_1 = loss(Q_vals_q1, Q_target)
            critic_loss_2 = loss(Q_vals_q2, Q_target)

            # Critic step Update
            self.critic_q1.train()
            self.critic_optimizer_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer_1.step()

            self.critic_q2.train()
            self.critic_optimizer_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optimizer_2.step()

            # TD3 updates the policy (and target networks) less frequently than the Q-function
            if self.update_counter % self.policy_freq_update == 0:

                # ------- calculate the actor loss
                actor_loss = - self.critic_q1.forward(states, self.actor.forward(states)).mean()

                self.actor.train()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # ------------------------------------- Update target networks --------------- #

                # update the target networks using tao "soft updates"
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target_q1.parameters(), self.critic_q1.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target_q2.parameters(), self.critic_q2.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def transition_model_learn(self):

        if self.memory.len_env_buffer() <= self.batch_size_model:
           return

        else:

            states, actions, _, next_states, _ = self.memory.sample_experience_from_env(self.batch_size_model)

            states      = np.array(states)  # (batch size, 16)
            actions     = np.array(actions)
            next_states = np.array(next_states)

            # Remove the target angle from state and next state because there is no point in predict
            # the target value since it randomly changes in the environment.
            states      = states[:, :-1]  # (sample_batch, 15 values)
            next_states = next_states[:, :-1]


            states      = torch.FloatTensor(states)       # torch.Size([sample_batch, 15])
            actions     = torch.FloatTensor(actions)      # torch.Size([sample_batch, 4])
            next_states = torch.FloatTensor(next_states)  # torch.Size([sample_batch, 15])

            distribution_probability_model_1 = self.pdf_transition_model_1.forward(states, actions)
            distribution_probability_model_2 = self.pdf_transition_model_2.forward(states, actions)
            distribution_probability_model_3 = self.pdf_transition_model_3.forward(states, actions)
            distribution_probability_model_4 = self.pdf_transition_model_4.forward(states, actions)
            distribution_probability_model_5 = self.pdf_transition_model_5.forward(states, actions)

            # calculate the loss
            loss_neg_log_likelihood_1 = - distribution_probability_model_1.log_prob(next_states)
            loss_neg_log_likelihood_2 = - distribution_probability_model_2.log_prob(next_states)
            loss_neg_log_likelihood_3 = - distribution_probability_model_3.log_prob(next_states)
            loss_neg_log_likelihood_4 = - distribution_probability_model_4.log_prob(next_states)
            loss_neg_log_likelihood_5 = - distribution_probability_model_5.log_prob(next_states)

            loss_neg_log_likelihood_1 = torch.mean(loss_neg_log_likelihood_1)
            loss_neg_log_likelihood_2 = torch.mean(loss_neg_log_likelihood_2)
            loss_neg_log_likelihood_3 = torch.mean(loss_neg_log_likelihood_3)
            loss_neg_log_likelihood_4 = torch.mean(loss_neg_log_likelihood_4)
            loss_neg_log_likelihood_5 = torch.mean(loss_neg_log_likelihood_5)

            self.pdf_transition_model_1.train()
            self.pdf_transition_1_optimizer.zero_grad()
            loss_neg_log_likelihood_1.backward()
            self.pdf_transition_1_optimizer.step()

            self.pdf_transition_model_2.train()
            self.pdf_transition_2_optimizer.zero_grad()
            loss_neg_log_likelihood_2.backward()
            self.pdf_transition_2_optimizer.step()

            self.pdf_transition_model_3.train()
            self.pdf_transition_3_optimizer.zero_grad()
            loss_neg_log_likelihood_3.backward()
            self.pdf_transition_3_optimizer.step()

            self.pdf_transition_model_4.train()
            self.pdf_transition_4_optimizer.zero_grad()
            loss_neg_log_likelihood_4.backward()
            self.pdf_transition_4_optimizer.step()

            self.pdf_transition_model_5.train()
            self.pdf_transition_5_optimizer.zero_grad()
            loss_neg_log_likelihood_5.backward()
            self.pdf_transition_5_optimizer.step()

            print("Loss:", loss_neg_log_likelihood_1.item(), loss_neg_log_likelihood_2.item(),
                           loss_neg_log_likelihood_3.item(), loss_neg_log_likelihood_4.item(),
                           loss_neg_log_likelihood_5.item())

    def generate_dream_samples(self):

        if self.memory.len_env_buffer() <= self.batch_size_model:
           return

        else:
            for _ in range(1, self.M + 1):

                state, _, _, _, _ = self.memory.sample_experience_from_env(batch_size=self.batch_size_model)

                state        = np.array(state)           # --> (sample_batch, 16)
                state_tensor = torch.FloatTensor(state)  # torch.Size([sample_batch, 16])

                target_angle = state[:, -1:]  # just target point only --> (sample_batch, 1)

                # Remove the target point from state
                state_input  = state[:, :-1]                   # (sample_batch, 15 values)
                state_input  = torch.FloatTensor(state_input)  # torch.Size([sample_batch, 15])

                # generate sample batch actions from policy

                with torch.no_grad():
                    self.actor.eval()
                    action = self.actor.forward(state_tensor)  # this takes the 16 values tensor
                    action = action.detach().numpy()
                    #action = action + (np.random.normal(0, scale=0.1, size=4))
                    action = np.clip(action, -1, 1)


                action_tensor = torch.FloatTensor(action)  # torch.Size([sample_batch, 4])

                # predict and generate new "dream" samples

                with torch.no_grad():
                    self.pdf_transition_model_1.eval()
                    function_generated_1 = self.pdf_transition_model_1.forward(state_input, action_tensor)
                    predicted_state_1    = function_generated_1.sample()  # torch.Size([sample_batch, 15])
                    predicted_state_1    = predicted_state_1.detach().numpy()  # (32, 15)

                self.pdf_transition_model_2.eval()
                with torch.no_grad():
                    function_generated_2 = self.pdf_transition_model_2.forward(state_input, action_tensor)
                    predicted_state_2    = function_generated_2.sample()  # torch.Size([sample_batch, 15])
                    predicted_state_2    = predicted_state_2.detach().numpy()  # (32, 15)

                self.pdf_transition_model_3.eval()
                with torch.no_grad():
                    function_generated_3 = self.pdf_transition_model_3.forward(state_input, action_tensor)
                    predicted_state_3    = function_generated_3.sample()  # torch.Size([sample_batch, 15])
                    predicted_state_3    = predicted_state_3.detach().numpy()  # (32, 15)

                self.pdf_transition_model_4.eval()
                with torch.no_grad():
                    function_generated_4 = self.pdf_transition_model_4.forward(state_input, action_tensor)
                    predicted_state_4    = function_generated_4.sample()  # torch.Size([sample_batch, 15])
                    predicted_state_4    = predicted_state_4.detach().numpy()  # (32, 15)

                self.pdf_transition_model_5.eval()
                with torch.no_grad():
                    function_generated_5 = self.pdf_transition_model_5.forward(state_input, action_tensor)
                    predicted_state_5    = function_generated_5.sample()  # torch.Size([sample_batch, 15])
                    predicted_state_5    = predicted_state_5.detach().numpy()  # (32, 15)


                next_state_imagined = np.mean(np.array([predicted_state_1, predicted_state_2, predicted_state_3,
                                                        predicted_state_4, predicted_state_5]), axis=0)

                # calculate the reward based on the prediction and input
                cylinder_angle   = next_state_imagined[:, -1:]  # (32, 1)
                target_position  = target_angle


                imagined_difference_cylinder_goal = np.abs(cylinder_angle - target_position)  # (32, 1)

                for single_state, single_action, single_next_state, single_distance_dif, single_target in zip(state, action, next_state_imagined, imagined_difference_cylinder_goal, target_position):

                    if single_distance_dif <= 10:
                        done     = True
                        reward_d = np.float64(1000)
                    else:
                        done = False
                        reward_d = -single_distance_dif[0]

                    full_next_state = np.append(single_next_state, single_target)  # (16,)

                    state_to_save      = single_state
                    next_state_to_save = full_next_state
                    action_to_save     = single_action
                    reward_to_save     = reward_d

                    self.add_imagined_experience_memory(state_to_save, action_to_save, reward_to_save, next_state_to_save, done)

    def save_rl_models(self):
        torch.save(self.actor.state_dict(),     'models/TD3/MBRL_TD3_actor_cylinder.pth')
        torch.save(self.critic_q1.state_dict(), 'models/TD3/MBRL_TD3_critic_1_cylinder.pth')
        torch.save(self.critic_q2.state_dict(), 'models/TD3/MBRL_TD3_critic_2_cylinder.pth')
        print("models has been saved...")

    def load_rl_models(self):
        self.actor.load_state_dict(torch.load(f"models/TD3/MBRL_TD3_actor_cylinder.pth"))
        self.critic_q1.load_state_dict(torch.load(f'models/TD3/MBRL_TD3_critic_1_cylinder.pth'))
        self.critic_q2.load_state_dict(torch.load(f'models/TD3/MBRL_TD3_critic_2_cylinder.pth'))
        print(f"models has been loaded...")

    def save_transition_models(self):
        torch.save(self.pdf_transition_model_1.state_dict(), f"models/TD3/TD3_transition_model_1.pth")
        torch.save(self.pdf_transition_model_2.state_dict(), f"models/TD3/TD3_transition_model_2.pth")
        torch.save(self.pdf_transition_model_3.state_dict(), f"models/TD3/TD3_transition_model_3.pth")
        torch.save(self.pdf_transition_model_4.state_dict(), f"models/TD3/TD3_transition_model_4.pth")
        torch.save(self.pdf_transition_model_5.state_dict(), f"models/TD3/TD3_transition_model_5.pth")

        print(f"models for transitions has been saved...")

    def load_transition_models(self):
        #self.pdf_transition_model_1.load_state_dict(torch.load(f"models/TD3/TD3_transition_model_1.pth"))
        #self.pdf_transition_model_2.load_state_dict(torch.load(f"models/TD3/TD3_transition_model_2.pth"))
        #self.pdf_transition_model_3.load_state_dict(torch.load(f"models/TD3/TD3_transition_model_3.pth"))
        #self.pdf_transition_model_4.load_state_dict(torch.load(f"models/TD3/TD3_transition_model_4.pth"))
        #self.pdf_transition_model_5.load_state_dict(torch.load(f"models/TD3/TD3_transition_model_5.pth"))

        self.pdf_transition_model_1.load_state_dict(torch.load(f"/home/anyone/Desktop/model_rotation_trained_middle/TD3_transition_model_1.pth"))
        self.pdf_transition_model_2.load_state_dict(torch.load(f"/home/anyone/Desktop/model_rotation_trained_middle/TD3_transition_model_2.pth"))
        self.pdf_transition_model_3.load_state_dict(torch.load(f"/home/anyone/Desktop/model_rotation_trained_middle/TD3_transition_model_3.pth"))
        self.pdf_transition_model_4.load_state_dict(torch.load(f"/home/anyone/Desktop/model_rotation_trained_middle/TD3_transition_model_4.pth"))
        self.pdf_transition_model_5.load_state_dict(torch.load(f"/home/anyone/Desktop/model_rotation_trained_middle/TD3_transition_model_5.pth"))
        print(f"transitions models loaded...")


def plot_reward_curves(rewards, avg_rewards, number=1):

    np.savetxt(f'result/TD3/TD3_rewards_MBRL.txt', rewards)
    np.savetxt(f'result/TD3/TD3_avg_reward_MBRL.txt', avg_rewards)

    plt.figure(number, figsize=(20, 10))
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"result/TD3/TD3_MBRL.png")
    #plt.show()
    print("training curve has been saved...")


def run_exploration(env, episodes, horizont, agent):
    mode = "Exploration"
    for episode in range(1, episodes+1):
        env.reset_env()
        print("reset goodd")

        for step in range(1, horizont+1):
            state, _ = env.state_space_function()
            action   = env.generate_sample_act()
            env.env_step(action)
            next_state, image_state = env.state_space_function()
            reward, done = env.calculate_reward()
            agent.add_real_experience_memory(state, action, reward, next_state, done)
            env.env_render(image=image_state, episode=episode, step=step, done=done, mode=mode, cylinder=next_state[-2:-1])
            if done:
                break
    print(f"******* -----{episodes} for exploration ended-----********* ")



def run_MB_training(env, episodes, horizont, agent):

    mode        = f"Training TD3 MBRL"
    rewards     = []
    avg_rewards = []

    for episode in range(1, episodes+1):

        env.reset_env()
        episode_reward = 0

        for step in range(1, horizont+1):

            state, _ = env.state_space_function()
            action   = agent.get_action_from_policy(state)
            noise    = np.random.normal(0, scale=0.15, size=4)
            action   = action + noise
            action   = np.clip(action, -1, 1)

            env.env_step(action)

            next_state, image_state = env.state_space_function()
            reward, done    = env.calculate_reward()
            episode_reward += reward
            agent.add_real_experience_memory(state, action, reward, next_state, done)
            env.env_render(image=image_state, episode=episode, step=step, done=done, mode=mode, cylinder=next_state[-2:-1])

            agent.transition_model_learn()
            agent.generate_dream_samples()

            if episode <= 10: # todo increase this number just put it 10 for testing
                data_type = 'env_data'
                print('working with env data')
                agent.step_training(data_type)
            else:
                data_type = 'model_data'
                print("working with model data")
                agent.step_training(data_type)

            if done:
                break

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

        print("Episode total reward:", episode_reward)
        print(f"******* -----Episode {episode} Ended-----********* ")

        if episode % 100 == 0:
            plot_reward_curves(rewards, avg_rewards, number=1)
            agent.save_transition_models()
            agent.save_rl_models()

    agent.save_transition_models()
    agent.save_rl_models()
    plot_reward_curves(rewards, avg_rewards, number=1)

    print(f"******* -----{episodes} episodes for training ended-----********* ")


def evaluate_model_transition(agent, env):

    agent.load_transition_models()

    for episode in range(1, 100):

        env.reset_env()

        for step in range(1, 25):

            state, _ = env.state_space_function()
            action   = env.generate_sample_act()
            env.env_step(action)
            next_state, _ = env.state_space_function()

            state        = np.array(state)
            state_input  = state[:-1]  # Remove the target point from state
            state_tensor = torch.FloatTensor(state_input)
            state_tensor = state_tensor.unsqueeze(0)  # torch.Size([1, 15])

            action_tensor = torch.FloatTensor(action)
            action_tensor = action_tensor.unsqueeze(0)  # torch.Size([1, 4])


            with torch.no_grad():
                agent.pdf_transition_model_1.eval()
                function_generated_1 = agent.pdf_transition_model_1.forward(state_tensor, action_tensor)
                predicted_state_1    = function_generated_1.sample()
                predicted_state_1    = predicted_state_1.detach().numpy()

            agent.pdf_transition_model_2.eval()
            with torch.no_grad():
                function_generated_2 = agent.pdf_transition_model_2.forward(state_tensor, action_tensor)
                predicted_state_2    = function_generated_2.sample()
                predicted_state_2    = predicted_state_2.detach().numpy()

            agent.pdf_transition_model_3.eval()
            with torch.no_grad():
                function_generated_3 = agent.pdf_transition_model_3.forward(state_tensor, action_tensor)
                predicted_state_3    = function_generated_3.sample()
                predicted_state_3    = predicted_state_3.detach().numpy()

            agent.pdf_transition_model_4.eval()
            with torch.no_grad():
                function_generated_4 = agent.pdf_transition_model_4.forward(state_tensor, action_tensor)
                predicted_state_4    = function_generated_4.sample()
                predicted_state_4    = predicted_state_4.detach().numpy()

            agent.pdf_transition_model_5.eval()
            with torch.no_grad():
                function_generated_5 = agent.pdf_transition_model_5.forward(state_tensor, action_tensor)
                predicted_state_5    = function_generated_5.sample()
                predicted_state_5    = predicted_state_5.detach().numpy()


            next_state_predicted = np.mean(np.array([predicted_state_1[0], predicted_state_2[0], predicted_state_3[0],
                                                    predicted_state_4[0], predicted_state_5[0]]), axis=0)

            print(next_state_predicted)
            print(next_state[:-1])
            print("error:")
            print(next_state[:-1] - next_state_predicted)
            print("-------------")
            


def main_run():

    env   = RL_ENV()
    agent = TD3agent_rotation_mbrl(env)

    num_exploration_episodes = 100
    num_episodes_training    = 1_000
    episode_horizont         = 20

    #run_exploration(env, num_exploration_episodes, episode_horizont, agent)
    #run_MB_training(env, num_episodes_training, episode_horizont, agent)
    #run_MB_test_mode(env, episode_horizont, agent)
    evaluate_model_transition(agent, env)


if __name__ == '__main__':
    main_run()