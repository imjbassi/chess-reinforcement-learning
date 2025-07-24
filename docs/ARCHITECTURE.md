# Chess-RL Architecture

## Overview

This document describes the architecture of the Chess-RL system.

## Components

### C++ Chess Engine

The C++ chess engine provides fast move generation and board evaluation:

- `board.h/cpp`: Chess board representation
- `movegen.h/cpp`: Move generation
- `attack_tables.h/cpp`: Precomputed attack tables
- `bindings.cpp`: Python bindings

### Neural Network

The neural network architecture is based on AlphaZero:

- Policy head: Predicts move probabilities
- Value head: Evaluates board positions

### Training Pipeline

The training process follows these steps:

1. Self-play to generate training data
2. Neural network training on the generated data
3. Evaluation against previous versions