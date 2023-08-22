"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import pandas as pd
from RL_brain import QLearningTable
import gym


def update():
    for episode in range(100):
        # initial state
        state, info = env.reset()
        steps = 0
        # done = False
        result = 'truncated'
        while True:
            steps += 1
            # fresh env
            env.render()

            # RL choose action based on state
            action = RL.choose_action(state)

            # RL take action and get next state and reward
            next_state, reward, terminated, truncated, info = env.step(action)
            print(f'state: {state}, action: {action}, next_state: {next_state}, reward: {reward}')
            
            # ------------
            if next_state == 10:
                reward = 1
                result = 'win'
            elif next_state == 6 or next_state == 9:
                reward = -1
                result = 'dead'
            else:
                reward = 0
            # -------------
            # RL learn from this transition
            qtable = RL.learn(state, action, reward, next_state, terminated)

            state = next_state

            # break while loop when end of this episode
            if terminated or truncated:
                print(f'------------ episode = {episode}, steps = {steps}, result = {result} ----------')
                print(qtable.sort_index())
                break

    # end of game
    # qtable.to_csv('q-table.csv', index=True)
    print('game over')


if __name__ == "__main__":
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    # --- gym version ---
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
        render_mode="human",
        map_name='4x4',
        desc=[
            "SFFF",
            "FFHF",
            "FHGF",
            "FFFF",
        ],
    )
    # env = env.unwrapped
    RL = QLearningTable(
        actions=list(range(env.action_space.n)),
        learning_rate=0.1,
    )
    update()
    env.close()
