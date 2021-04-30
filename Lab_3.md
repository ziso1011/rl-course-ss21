# Lab 3

In this lab we will continue our work on the [FrozenLake environment](https://gym.openai.com/envs/FrozenLake-v0/)
by introducing MC-Control and Q-learning.


### Task 1:
In the last lab we calculated Q-values using MC prediction. We will now extend this code to implement MC control
as we have seen it in the lecture:

- Take the code from the file [3_FrozenLake_Control.py](https://github.com/pabair/rl-course-ss21/blob/main/3_FrozenLake_Control.py) as starting point.
This code uses a random policy and plots the collected rewards over time.
- Integrate the code for calculating Q-values after every episode [from the last lab](https://github.com/pabair/rl-course-ss21/blob/main/solutions/S2_FrozenLake_Prediction_Task1.py).
- Change the `play_episode` method such that it uses an epsilon-greedy policy based on the current Q-values.
- Try out the following epsilons: `[0.01, 0.1, 0.5, 1.0]` and show all results for all epsilons together in one plot (i.e. every epsilon one curve in the plot).

### Task 2:
Implement now Q-learning as comparison control strategy:

- Redo task 1 using Q-Learning instead of MC control (use `alpha=0.5`).
- As above, try out the different epsilons and compare them in one plot.
