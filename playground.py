# https://github.com/openai/gym/wiki/Leaderboard

import gym
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

def q_model(n_input, n_output, learning_rate):
    model = Sequential()
    model.add(Dense(24, input_shape=(n_input,), activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(n_output, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return model

class DDPG(object):
    """docstring for DDPG"""
    def __init__(self, arg):
        super(DDPG, self).__init__()
        self.arg = arg

class DQN(object):
    """
    Value based method.
    Use deep nerual network arrpx. Q(s, a)
    """
    def __init__(self, env):
        self.GAMMA = 0.95
        self.LEARNING_RATE = 0.001
        self.MEMORY_SIZE = 1000000
        self.BATCH_SIZE = 20
        self.EXPLORATION_MAX = 1.0
        self.EXPLORATION_MIN = 0.01
        self.EXPLORATION_DECAY = 0.995
        self.n_act = env.action_space.n # 2
        self.n_ob = env.observation_space.shape[0] # 4
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.model = q_model(self.n_ob, self.n_act, self.LEARNING_RATE)

    def experience_replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = random.sample(list(self.memory), self.BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = reward + self.GAMMA * np.amax(self.model.predict(state_next)[0])
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.EXPLORATION_MAX *= self.EXPLORATION_DECAY
        self.EXPLORATION_MAX = max(self.EXPLORATION_MIN, self.EXPLORATION_MAX)

    def remember(self, state, action, reward, state_next, done):
        state = np.reshape(state, [1, self.n_ob])
        state_next = np.reshape(state_next, [1, self.n_ob])
        self.memory.append((state, action, reward, state_next, done))

    def act(self, state):
        state = np.reshape(state, [1, self.n_ob])
        if np.random.rand() < self.EXPLORATION_MAX:
            return random.randrange(self.n_act)
        q_values = self.model.predict(state) # [[22.010157 22.687887]]
        return np.argmax(q_values[0])

    def run_begin(self):
        pass

    def run_finish(self):
        pass
        
class DoubleDQN(DQN):
    """
    Use double parameters for Q-Network
    TODO 
    """
    def __init__(self, env):
        super(DoubleDQN, self).__init__(env)
        self.model1 = q_model(self.n_ob, self.n_act, self.LEARNING_RATE)
        self.model2 = q_model(self.n_ob, self.n_act, self.LEARNING_RATE)

    def switch_model(self):
        tmp = self.model1
        self.model1 = self.model2
        self.model2 = tmp

    def experience_replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = random.sample (list(self.memory), self.BATCH_SIZE)
        self.switch_model()
        for state, action, reward, state_next, terminal in batch:
            q_update = reward 
            if not terminal:
                a_argmax = np.argmax(self.model2.predict(state_next))
                q_update = reward + self.GAMMA * self.model1.predict(state_next)[0][a_argmax]
            q_values = self.model1.predict(state)
            q_values[0][action] = q_update
            self.model1.fit(state, q_values, verbose=0)
        self.EXPLORATION_MAX *= self.EXPLORATION_DECAY
        self.EXPLORATION_MAX = max(self.EXPLORATION_MIN, self.EXPLORATION_MAX)

    def act(self, state):
        state = np.reshape(state, [1, self.n_ob])
        if np.random.rand() < self.EXPLORATION_MAX:
            return random.randrange(self.n_act)
        q_values = self.model2.predict(state) # [[22.010157 22.687887]]
        return np.argmax(q_values[0])

class QLearningTabular(object):
    """
    Value-based method.
    Use tabular method to estimate Q(s, a)
    """
    def __init__(self, env):
        super(QLearningTabular, self).__init__()
        self.EPSILON = 0.05 # explore prob
        self.GAMMA = 0.5 # discount coef
        self.LEARNING_RATE = 1.0 # learning rate
        self.env = env
        self.q = {}
        self.max_steps = []
        self.round_bit = 1
        self.n_act = env.action_space.n # 2
        self.n_ob = env.observation_space.shape[0] # 4
    
    def run_begin(self):
        self.S = []
        self.A = []
        self.cnt_rand = 0
        self.cnt_tabu = 0
    
    def get_a_of_max_q(self, s):
        a_q_li = sorted(list(self.q[s].items()), key=lambda x:x[1], reverse=True)[0]
        return a_q_li[0]
    
    def get_max_q(self, s):
        a_q_li = sorted(list(self.q[s].items()), key=lambda x:x[1], reverse=True)[0]
        return a_q_li[1]
    
    def act(self, state):
        a = None
        # tabular lookup
        s = str(state.round(self.round_bit))
        if s in self.q:
            # pick highest score a
            a = self.get_a_of_max_q(s)
            self.cnt_tabu += 1
        # random
        if a is None or random.random() < self.EPSILON:
            a = self.env.action_space.sample()
            self.cnt_rand += 1
        return a
    
    def qlearning(self):
        n = len(self.A) - 1 
        for i in range(n):
            s = self.S[i]
            s_next = self.S[i + 1]
            a = self.A[i]
            # init with zero
            self.q.setdefault(s, {})
            self.q.setdefault(s_next, {})
            for each_a in range(self.n_act):
                self.q[s].setdefault(each_a, 0)
                self.q[s_next].setdefault(each_a, 0)
            reward = (n - i)
            # V1 Simple update: 35 mean_steps_last_1k @ 2k epoch
            # self.q[s][a] = max(self.q[s][a], reward)
            # V2 Q-learning update: 52 mean steps @ 2k epoch
            #    Q(s,a) <- Q(s,a) + LEARNING_RATE * [reward + GAMMA * max_a' Q(s',a')  - Q(s,a)]
            self.q[s][a] = self.q[s][a] + self.LEARNING_RATE * (reward + self.GAMMA * self.get_max_q(s_next) - self.q[s][a])
    
    def remember(self, state, action, reward, state_next, terminal):
        state = str(state.round(self.round_bit))
        self.A.append(action)
        self.S.append(state)
    
    def run_finish(self):
        assert len(self.A) == len(self.S)
        self.max_steps.append(len(self.S))
        print(f'rand_step:{self.cnt_rand / (self.cnt_rand + self.cnt_tabu) * 100:.2f}% mean_steps_last_1k:{np.mean(self.max_steps[-1000:]):.0f}')
        self.qlearning()
    
    def experience_replay(self):
        pass

def main():
    n_episode = 50
    for method in ['DoubleDQN']:
        env = gym.make(ENV_NAME)
        agent = None
        if method == 'DDPG':
            agent = DDPG(env)
        elif method == 'QLeaning':
            agent = QLearningTabular(env)
        elif method == 'DQN':
            agent = DQN(env)
        elif method == 'DoubleDQN':
            agent = DoubleDQN(env)
        else:
            raise NotImplementedError
        score_logger = ScoreLogger(ENV_NAME, method)
        print('Algorithm:', method)
        
        i_episode = 0
        while i_episode < n_episode:
            i_episode += 1
            agent.run_begin()
            state = env.reset()
            i_step = 0
            while True:
                i_step += 1
                # env.render()
                action = agent.act(state)
                state_next, reward, terminal, info = env.step(action)
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()
                state = state_next
                if terminal:
                    agent.run_finish()
                    print(f"Run {i_episode} finished after {i_step + 1} steps")
                    score_logger.add_score(i_step, i_episode)
                    break
        env.close()

if __name__ == '__main__':
    main()
