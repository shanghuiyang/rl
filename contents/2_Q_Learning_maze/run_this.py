import pandas as pd
from q_learing import QLearning
import gym

MAX_EPISODE = 200

pd.set_option('display.float_format', lambda x: '%.8f' % x)
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
ql = QLearning(
    actions=list(range(env.action_space.n)),
    learning_rate=0.1,
    load_qtable_from_csv=False,
    save_qtable_to_csv=False,
)

# training
for episode in range(MAX_EPISODE):
    steps = 0
    result = 'truncated'
    terminated, truncated = False, False
    state, info = env.reset()
    while not terminated and not truncated:
        steps += 1
        env.render()
        action = ql.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f'state: {state}, action: {action}, next_state: {next_state}, reward: {reward}')

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

        qtable = ql.learn(state, action, reward, next_state, terminated)
        state = next_state

    print(f'------- episode: {episode+1}, steps: {steps}, result: {result} --------')
    print(qtable.sort_index())
    print('--------------------------------------------------\n')


ql.save_qtable()
env.close()
print('training done')
