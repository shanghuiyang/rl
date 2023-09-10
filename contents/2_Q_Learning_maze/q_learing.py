import numpy as np
import pandas as pd

Q_TABLE_CSV = 'q-table.csv'


class QLearning:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, load_qtable_from_csv=False, save_qtable_to_csv=False):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.load_qtable_from_csv = load_qtable_from_csv
        self.save_qtable_to_csv = save_qtable_to_csv
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        if load_qtable_from_csv:
            print(f'loading q-table from ./{Q_TABLE_CSV}')
            df = pd.read_csv(Q_TABLE_CSV, index_col=0)
            df.columns = df.columns.astype(int)
            self.q_table = df
            print('------------------ q-table ------------------')
            print(self.q_table)
            print('---------------------------------------------')

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            # exploit
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # explore
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, terminated):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if not terminated:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
        return self.q_table

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            new_row = pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table.loc[state] = new_row

    def save_qtable(self):
        if not self.save_qtable_to_csv:
            return
        self.q_table.sort_index(inplace=True)
        self.q_table.to_csv(Q_TABLE_CSV, index=True)
