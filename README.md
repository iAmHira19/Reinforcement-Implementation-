# Reinforcement Learning with Q-learning

This project demonstrates the implementation of a simple Q-learning algorithm using the OpenAI Gym environment. The example uses the "FrozenLake" environment to train an agent to navigate a grid to reach a goal.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Training the Agent](#training-the-agent)
- [Evaluating the Agent](#evaluating-the-agent)
- [Visualizing the Agent's Performance](#visualizing-the-agents-performance)
- [Conclusion](#conclusion)

## Overview

Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. This project implements the Q-learning algorithm, a popular reinforcement learning algorithm that uses a Q-table to store and update the expected rewards for state-action pairs.

## Setup

To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required dependencies:
    ```sh
    pip install gym
    ```

## Training the Agent

The Q-learning algorithm is used to train the agent. The algorithm updates the Q-values based on the agent's actions and the rewards received from the environment. The following hyperparameters are used:
- `alpha` (learning rate): 0.1
- `gamma` (discount factor): 0.6
- `epsilon` (exploration rate): 0.1

```python
# Training the agent
num_episodes = 1000

for i in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values
        
        next_state, reward, done, info = env.step(action)
        
        # Update Q-Table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        
        state = next_state

print("Training finished.\n")
```

## Evaluating the Agent

After training, the agent is evaluated over a number of episodes to measure its performance.

```python
# Evaluate the agent
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        
        if reward == 0:
            penalties += 1
        
        epochs += 1
    
    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
```

## Visualizing the Agent's Performance

The trained agent's performance can be visualized by rendering the environment and displaying the agent's actions.

```python
# Visualize the agent's performance
state = env.reset()
done = False

for _ in range(100):
    clear_output(wait=True)
    env.render()
    sleep(1)
    
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)
    
    if done:
        clear_output(wait=True)
        env.render()
        sleep(1)
        break
```

## Conclusion

This project demonstrates a basic implementation of the Q-learning algorithm using the OpenAI Gym "FrozenLake" environment. The agent learns to navigate the grid and reach the goal by updating its Q-values based on the rewards received. This project can be extended to more complex environments and algorithms to further explore the capabilities of reinforcement learning.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
