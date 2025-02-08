# Tetris AI with DQN (Deep Q-Learning)

This is a simple implementation of a Tetris game with a reinforcement learning agent using Deep Q-Learning (DQN). The agent learns how to play Tetris by interacting with the environment and optimizing its actions based on rewards.

## Project Description

This project involves training a Deep Q-Network (DQN) to play the Tetris game. The game environment is simulated using `pygame`, and the neural network model is built with PyTorch. The agent learns to make decisions such as moving left, right, rotating, and dropping pieces to clear lines and gain points.

### Key Features:
- **Tetris Game Environment**: Implemented using `pygame` for visual rendering.
- **Deep Q-Learning Agent**: The agent uses a convolutional neural network (CNN) to predict Q-values for each action.
- **Rewards and Training**: The agent is rewarded for clearing lines and punished when the game is over.

### Prerequisites:
- Python 3.10
- pip (Python package installer)
