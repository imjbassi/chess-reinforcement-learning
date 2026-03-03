# Chess-RL Architecture

## Overview

This document describes the architecture of the Chess-RL system, a reinforcement learning framework for training chess-playing agents using self-play and neural networks.

## System Architecture

The system consists of three main layers:
1. **C++ Chess Engine** - High-performance move generation and board representation
2. **Neural Network** - Deep learning model for position evaluation and move selection
3. **Training Pipeline** - Self-play and learning infrastructure

## Components

### C++ Chess Engine

The C++ chess engine provides fast, efficient move generation and board evaluation. It is designed for performance-critical operations during self-play.

#### Core Modules

- **`board.h/cpp`**: Chess board representation using bitboards for efficient state management
  - Maintains piece positions, castling rights, en passant squares
  - Provides methods for making/unmaking moves
  - Implements Zobrist hashing for position identification

- **`movegen.h/cpp`**: Legal move generation
  - Generates pseudo-legal moves for all piece types
  - Validates moves for legality (king safety checks)
  - Optimized for speed using bitboard operations

- **`attack_tables.h/cpp`**: Precomputed attack tables
  - Magic bitboards for sliding piece attacks (rooks, bishops, queens)
  - Lookup tables for knight and king moves
  - Initialized once at startup for fast runtime queries

- **`bindings.cpp`**: Python bindings using pybind11
  - Exposes C++ functionality to Python training code
  - Provides efficient data transfer between C++ and Python
  - Maintains minimal overhead for cross-language calls

### Neural Network

The neural network architecture is inspired by AlphaZero, using a shared trunk with dual output heads.

#### Architecture Details

- **Input Representation**: Board state encoded as feature planes
  - Piece positions (12 planes: 6 piece types × 2 colors)
  - Auxiliary features (castling rights, en passant, move count)

- **Shared Trunk**: Convolutional residual network
  - Multiple residual blocks for feature extraction
  - Batch normalization and ReLU activations
  - Processes spatial relationships on the board

- **Policy Head**: Predicts move probabilities
  - Outputs probability distribution over all legal moves
  - Used for move selection during self-play and inference
  - Trained to match improved policy from MCTS

- **Value Head**: Evaluates board positions
  - Outputs scalar value estimating win probability
  - Range: [-1, 1] where -1 = loss, 0 = draw, 1 = win
  - Trained on game outcomes from self-play

### Training Pipeline

The training process follows an iterative improvement cycle based on reinforcement learning principles.

#### Training Steps

1. **Self-Play Generation**
   - Current neural network plays games against itself
   - Monte Carlo Tree Search (MCTS) guides move selection
   - Generates training examples: (position, improved_policy, outcome)
   - Stores experience in a replay buffer

2. **Neural Network Training**
   - Samples mini-batches from the replay buffer
   - Trains network to predict MCTS policies and game outcomes
   - Uses combined loss: policy loss + value loss + regularization
   - Optimizes using stochastic gradient descent (SGD) or Adam

3. **Model Evaluation**
   - New model plays against previous best model
   - Determines if new model is an improvement
   - Promotes new model if win rate exceeds threshold (e.g., 55%)
   - Maintains version history for analysis

#### Data Flow

```
Self-Play → Experience Buffer → Training Batches → Network Update → Evaluation → Best Model
    ↑                                                                               |
    └───────────────────────────────────────────────────────────────────────────────┘
```

## Design Principles

- **Performance**: C++ engine for compute-intensive operations
- **Flexibility**: Python for experimentation and training logic
- **Modularity**: Clear separation between engine, network, and training
- **Scalability**: Designed for distributed self-play and training

## Future Enhancements

Potential areas for expansion:
- Distributed self-play across multiple machines
- Opening book integration
- Endgame tablebase support
- Multi-GPU training support
- Advanced MCTS variants (e.g., Gumbel AlphaZero)