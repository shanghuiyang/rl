import pandas as pd
from q_learing import QLearning
import gym

MAX_EPISODE = 2000

pd.set_option('display.float_format', lambda x: '%.8f' % x)
env = gym.make(
    "FrozenLake-v1",
    is_slippery=False,
    max_episode_steps=600,
    render_mode="human",  # ansi, human
    map_name='4x4',
    desc=[
        "SFFF",
        "FFHF",
        "FHGF",
        "FFFF",
    ],
    # map_name='8x8',
    # desc=[
    #     "SFFHFFFF",
    #     "FHFFHFHF",
    #     "FHFHFFHF",
    #     "HFFHFHFF",
    #     "FFHFFHFF",
    #     "HFHFFHFH",
    #     "HFFHFHFH",
    #     "FHFFFHFG",
    # ],
)
# env = env.unwrapped
ql = QLearning(
    actions=list(range(env.action_space.n)),
    learning_rate=0.1,
    e_greedy=0.99,
    load_qtable_from_csv=False,
    save_qtable_to_csv=False,
)

# training
for episode in range(MAX_EPISODE):
    steps = 0
    result = 'truncated'
    terminated, truncated = False, False
    done = (terminated | truncated)
    state, info = env.reset()

    while not done:
        steps += 1
        env.render()
        action = ql.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        if truncated:
            result = 'truncated'
            break

        # -- customed reward -------
        reached_the_gold = (terminated and reward == 1)
        fell_in_a_hole = (terminated and reward == 0)
        if reached_the_gold:
            reward = 1
            result = 'win'
        elif fell_in_a_hole:
            reward = -1
            result = 'miss'
        else:
            reward = 0
        # --------------------------

        print(f'state: {state}, action: {action}, next_state: {next_state}, reward: {reward}')
        qtable = ql.learn(state, action, reward, next_state, terminated)
        state = next_state
        done = (terminated | truncated)

    print(f'------- episode: {episode+1}, steps: {steps}, result: {result} --------')
    print(qtable.sort_index())
    print('--------------------------------------------------\n')


ql.save_qtable()
env.close()
print('training done')
