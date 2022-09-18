"""
    Task: Rotation Cylinder V2
    Algorithm: TD3- MFRL
    Version V2.0
    Task Learned ok
"""

from Memory   import MemoryClass
from Networks import Actor, Critic
from main_rl_env_rotation_v2 import RL_ENV

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


class TD3agent_rotation:
    def __init__(self, env):

        # -------- Hyper-parameters --------------- #
        self.env = env
        self.gamma      = 0.99  # discount factor
        self.tau        = 0.005
        self.batch_size = 32

        self.G = 15
        self.update_counter     = 0
        self.policy_freq_update = 2
        self.inter_act_counter  = 0

        self.max_memory_size = 20_000

        self.actor_learning_rate  = 1e-4
        self.critic_learning_rate = 1e-3

        self.hidden_size_critic = [128, 64, 32]
        self.hidden_size_actor  = [128, 64, 32]

        # -------- Parameters --------------- #
        self.num_states  = 16
        self.num_actions = 4

        # ------------- Initialization memory --------------------- #
        self.memory = MemoryClass(self.max_memory_size)

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


    def get_action_demostration(self):
        self.inter_act_counter += 1
        if self.inter_act_counter % 2 == 0:
            act_m1_demonstration = -1
            act_m2_demonstration = -1
            act_m3_demonstration = -1
            act_m4_demonstration = 1
            action_vector = np.array([act_m1_demonstration, act_m2_demonstration, act_m3_demonstration, act_m4_demonstration])
        else:
            act_m1_demonstration = 1
            act_m2_demonstration = 1
            act_m3_demonstration = 1
            act_m4_demonstration = -1
            action_vector = np.array([act_m1_demonstration, act_m2_demonstration, act_m3_demonstration, act_m4_demonstration])

        return action_vector

    def get_action_from_policy(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # numpy to a tensor with shape [1,15]
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state_tensor)
            action = action.detach()
            action = action.numpy()  # tensor to numpy
            self.actor.train()
        return action[0]

    def add_experience_memory(self, state, action, reward, next_state, done):
        # Save experience in memory
        self.memory.replay_buffer_add(state, action, reward, next_state, done)

    def step_training(self):
        # check, if enough samples are available in memory
        if self.memory.__len__() <= self.batch_size:
            return

        # update the networks every G times
        for it in range(1, self.G+1):

            self.update_counter += 1
            states, actions, rewards, next_states, dones = self.memory.sample_experience(self.batch_size)

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
            self.critic_optimizer_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer_1.step()

            self.critic_optimizer_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optimizer_2.step()

            # TD3 updates the policy (and target networks) less frequently than the Q-function
            if self.update_counter % self.policy_freq_update == 0:
                # ------- calculate the actor loss
                actor_loss = - self.critic_q1.forward(states, self.actor.forward(states)).mean()
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


    def save_model(self):
        torch.save(self.actor.state_dict(),     'models/TD3/MFRL_TD3_actor_cylinder.pth')
        torch.save(self.critic_q1.state_dict(), 'models/TD3/MFRL_TD3_critic_1_cylinder.pth')
        torch.save(self.critic_q2.state_dict(), 'models/TD3/MFRL_TD3_critic_2_cylinder.pth')
        print("models has been saved...")

    def load_model(self):
        self.actor.load_state_dict(torch.load('models/TD3/MFRL_TD3_actor_cylinder.pth'))
        self.critic_q1.load_state_dict(torch.load('models/TD3/MFRL_TD3_critic_1_cylinder.pth'))
        self.critic_q2.load_state_dict(torch.load('models/TD3/MFRL_TD3_critic_2_cylinder.pth'))
        print("models has been loaded...")




def plot_reward_curves(rewards, avg_rewards, number=1):

    np.savetxt(f'result/TD3/MFRL_rewards.txt', rewards)
    np.savetxt(f'result/TD3/MFRL_avg_reward.txt', avg_rewards)

    plt.figure(number, figsize=(20, 10))
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"result/TD3/MFRL_TD3.png")
    #plt.show()
    print("training curve has been saved...")


def run_test_mode(env, episode_horizont, agent):

    agent.load_model()
    mode        = f"Testing"
    rewards     = []
    episodes_test = 100

    for episode in range(1, episodes_test+1):

        env.reset_env()
        episode_reward = 0
        for step in range(episode_horizont):
            print(f"-------Episode:{episode + 1} Step:{step + 1}---------")
            state, _ = env.state_space_function()
            action   = agent.get_action_from_policy(state)
            env.env_step(action)
            next_state, image_state = env.state_space_function()
            reward, done    = env.calculate_reward()
            episode_reward += reward
            env.env_render(image=image_state, episode=episode, step=step, done=done, mode=mode, cylinder=next_state[-2:-1])
            if done:
                break
        print("Episode total reward:", episode_reward)
        rewards.append(episode_reward)

    plt.figure(3, figsize=(20, 10))
    plt.plot(rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


def run_exploration(env, episodes, horizont, agent):
    mode = "Exploration"
    for episode in range(1, episodes+1):
        env.reset_env()
        for step in range(1, horizont+1):
            state, _ = env.state_space_function()
            action   = env.generate_sample_act()
            env.env_step(action)
            next_state, image_state = env.state_space_function()
            reward, done = env.calculate_reward()
            agent.add_experience_memory(state, action, reward, next_state, done)
            env.env_render(image=image_state, episode=episode, step=step, done=done, mode=mode, cylinder=next_state[-2:-1])
            if done:
                break
    print(f"******* -----{episodes} for exploration ended-----********* ")


def run_training(env, num_episodes_training, episode_horizont, agent):
    mode = "Training TD3"

    rewards     = []
    avg_rewards = []

    for episode in range(1, num_episodes_training + 1):
        env.reset_env()
        episode_reward = 0

        for step in range(1, episode_horizont + 1):
            state, _ = env.state_space_function()

            action   = agent.get_action_from_policy(state)
            noise    = np.random.normal(0, scale=0.15, size=4)
            action   = action + noise
            action   = np.clip(action, -1, 1)

            env.env_step(action)

            next_state, image_state = env.state_space_function()
            reward, done = env.calculate_reward()
            episode_reward += reward
            agent.add_experience_memory(state, action, reward, next_state, done)
            env.env_render(image=image_state, episode=episode, step=step, done=done, mode=mode, cylinder=next_state[-2:-1])
            agent.step_training()
            if done:
                break

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-100:]))

        print(f"******* -----Episode {episode} Ended-----********* ")
        print("Episode total reward:", episode_reward)
        if episode % 100 == 0:
            agent.save_model()
            plot_reward_curves(rewards, avg_rewards, number=1)


    agent.save_model()
    plot_reward_curves(rewards, avg_rewards, number=1)


def main_run():

    env   = RL_ENV()
    agent = TD3agent_rotation(env)

    num_exploration_episodes = 500
    num_episodes_training    = 5_000
    episode_horizont         = 20

    run_exploration(env, num_exploration_episodes, episode_horizont, agent)
    run_training(env, num_episodes_training, episode_horizont, agent)
    #run_test_mode(env, episode_horizont, agent)



if __name__ == '__main__':
    main_run()

