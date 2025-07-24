# Chess-RL: Chess Reinforcement Learning Engine

A neural network-based chess engine that learns through self-play reinforcement learning.

## Features

- Fast C++ chess engine for move generation and board representation
- PyTorch neural network that learns from self-play
- Visualization and GUI using Pygame
- Python fallback implementation for reliability

## Project Structure

- `/chessengine/` - C++ chess engine implementation
- `/python/` - Python modules
  - `/gui/` - Pygame visualization
  - `/model/` - Neural network architecture
  - `/train/` - Training and self-play
  - `/engine/` - Python binding for C++ engine

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Pygame 2.0+
- CMake 3.10+
- C++ compiler with C++17 support

## Installation

```bash
# Clone the repository
git clone https://github.com/imjbassi/chess-rl.git
cd chess-rl

# Install Python dependencies
pip install -r requirements.txt

# Build C++ engine
mkdir build
cd build
cmake ..
make
```

## Usage

### Training the model
```bash
python python/train/train.py
```

### Self-play data generation
```bash
python python/train/selfplay.py
```

### GUI visualization
```bash
python python/gui/pygame_gui.py
```