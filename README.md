# Tetris AI with Deep Q-Learning (DQN)

This project implements a Tetris AI using Deep Q-Networks (DQN). MI

## Features

- Implements a **5x10** Tetris grid with **1x1** blocks.
- Uses **Deep Q-Learning (DQN)** for decision-making.
- Standard Tetris scoring system.
- The trained model can be saved and loaded for further use.
- Supports visualization of gameplay.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Game

```bash
python main.py
```

## Training the AI

To train the AI using DQN, run:

```bash
python train.py
```

## Saving and Loading the Model

The trained model is automatically saved as `tetris_model.pth`. To load the model and run the AI:

```bash
python play.py --load-model tetris_model.pth
```

## Project Structure

```
|-- tetris.py         # Tetris game logic
|-- dqn.py            # Deep Q-Network implementation
|-- train.py          # Training script
|-- play.py           # Running AI script
|-- utils.py          # Helper functions
|-- requirements.txt  # Dependencies
|-- README.md         # Project documentation
```

## Contributing

Feel free to open an issue or submit a pull request if you want to improve the project!

## License

MIT License

