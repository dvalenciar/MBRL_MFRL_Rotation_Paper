from collections import deque
import random

class MemoryClass:

    def __init__(self, replay_max_size):
        self.replay_max_size = replay_max_size
        self.replay_buffer = deque(maxlen=replay_max_size)  # buffer of experiences to sample during training

    def replay_buffer_add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.append(experience)

    def sample_experience(self, batch_size):
        state_batch      = []
        action_batch     = []
        reward_batch     = []
        next_state_batch = []
        done_batch       = []

        batch = random.sample(self.replay_buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.replay_buffer)