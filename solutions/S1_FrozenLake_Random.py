import gym
import random

env = gym.make("FrozenLake-v0", is_slippery=False)

random.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()

no_of_actions = env.env.nA
action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}


def play_episode(env, policy=None):
    state = env.reset()
    done = False
    total_reward = 0
    states = [state]
    actions = []
    while not done:
        if policy is  None:
            action = random.randint(0, no_of_actions-1)
        else:
            action = policy[state] # choose a random action

        actions.append(action)
        state, reward, done, _ = env.step(action)
        states.append(state)
        total_reward += reward

    return states, actions, total_reward


# Task 1:
print(f"\n ### TASK 1 ### ")

count = 0
while True:
    s, a, r = play_episode(env)
    count += 1
    if r > 0:
        break

print(f"Converged after {count} episodes.")
print(f"Random policy took {len(s)} steps.")

policy = {}
for i, v in enumerate(s[:-1]):
    policy[v] = a[i]
print("Improving policy to:", policy)

s, a, r = play_episode(env, policy)
if r > 0:
    print(f"Success: New policy took {len(s)} steps.")
else:
    print("New policy failed!")



