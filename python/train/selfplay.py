```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import traceback
import random
import threading

# Import our Python chess implementation
try:
    from python_chess import SimpleChessBoard, encode_simple_board
    print("Successfully imported python_chess module")
except ImportError as e:
    print(f"Error importing python_chess: {e}")
    traceback.print_exc()
    sys.exit(1)

# Try to import the C++ engine
try:
    from engine.chess_board import ChessBoard
    from model.model import encode_board, ChessNet
    CPP_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"Warning: C++ chess engine unavailable: {e}")
    print("Using Python-only chess implementation.")
    CPP_ENGINE_AVAILABLE = False
    from model.model import ChessNet


def get_moves_with_timeout(board, timeout=3):
    """
    Try to get legal moves with a timeout, fall back to Python if needed.
    
    Args:
        board: Chess board instance (SimpleChessBoard or ChessBoard)
        timeout: Maximum time in seconds to wait for move generation
        
    Returns:
        List of legal moves in UCI format
    """
    if not CPP_ENGINE_AVAILABLE:
        # If the C++ engine isn't available, use our Python implementation directly
        if isinstance(board, SimpleChessBoard):
            return board.get_legal_moves()
        else:
            print("Warning: C++ engine unavailable and board is not SimpleChessBoard")
            return []
    
    # Otherwise, use the timeout approach with the C++ engine
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = board.get_legal_moves()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    
    thread.join(timeout)
    if thread.is_alive():
        print("WARNING: get_legal_moves() timed out! Using Python fallback...")
        py_board = SimpleChessBoard()
        return py_board.get_legal_moves()
    
    if exception[0]:
        print(f"Exception in get_legal_moves: {exception[0]}")
        print("Using Python fallback...")
        py_board = SimpleChessBoard()
        return py_board.get_legal_moves()
    
    return result[0]


def _uci_to_index(uci):
    """
    Convert UCI move string to flat move index.
    
    Maps a move like "e2e4" to an index in [0, 4095] by treating
    the board as a 64x64 from-to matrix flattened into a vector.
    
    Args:
        uci: Move in UCI format (e.g., "e2e4")
        
    Returns:
        Integer index in range [0, 4095] representing the move
    """
    from_square = (ord(uci[0]) - ord('a')) + (ord(uci[1]) - ord('1')) * 8
    to_square = (ord(uci[2]) - ord('a')) + (ord(uci[3]) - ord('1')) * 8
    return from_square * 64 + to_square


def _index_to_uci(index):
    """
    Convert flat move index to UCI move string.
    
    Inverse of _uci_to_index, converts an index back to UCI notation.
    
    Args:
        index: Integer index in range [0, 4095]
        
    Returns:
        Move in UCI format (e.g., "e2e4")
    """
    from_square, to_square = divmod(index, 64)
    from_file = chr(from_square % 8 + ord('a'))
    from_rank = str(from_square // 8 + 1)
    to_file = chr(to_square % 8 + ord('a'))
    to_rank = str(to_square // 8 + 1)
    return f"{from_file}{from_rank}{to_file}{to_rank}"


def _select_move_from_policy(logits, legal_moves, temperature=1.0):
    """
    Select a move from the policy network output.
    
    Masks illegal moves, applies temperature scaling, and samples from
    the resulting probability distribution.
    
    Args:
        logits: Raw policy logits from the network (shape: [1, 4096])
        legal_moves: List of legal moves in UCI format
        temperature: Temperature parameter for exploration (higher = more random)
        
    Returns:
        Tuple of (selected_uci_move, probability_distribution)
    """
    if not legal_moves:
        raise ValueError("No legal moves available")
    
    # Create a mask for legal moves only
    mask = torch.full_like(logits, float('-inf'))
    for uci in legal_moves:
        idx = _uci_to_index(uci)
        mask[0, idx] = logits[0, idx]
    
    # Apply temperature for exploration/exploitation control
    if temperature > 0 and temperature != 1.0:
        mask = mask / temperature
    
    # Convert logits to probabilities
    probs = F.softmax(mask, dim=1)
    
    # Sample move from the probability distribution
    try:
        move_idx = torch.multinomial(probs, num_samples=1).item()
        uci = _index_to_uci(move_idx)
        
        # Verify move is legal
        if uci not in legal_moves:
            raise ValueError(f"Selected move {uci} not in legal moves")
            
    except (ValueError, RuntimeError) as e:
        # Fallback to random move if sampling fails
        print(f"Error sampling from policy: {e}")
        uci = random.choice(legal_moves)
        print(f"Falling back to random move: {uci}")
        
        # Update probabilities for this move
        probs = torch.zeros_like(probs)
        idx = _uci_to_index(uci)
        probs[0, idx] = 1.0
    
    return uci, probs


def selfplay(net, n_games=10, max_plies=200, temperature=1.0):
    """
    Run self-play games using neural network policy for move selection.
    
    Generates training data by having the network play against itself.
    Each position is stored along with the policy used and the final outcome.
    
    Args:
        net: ChessNet neural network model
        n_games: Number of games to play
        max_plies: Maximum number of half-moves per game
        temperature: Temperature parameter for move selection (higher = more exploration)
        
    Returns:
        List of training examples as (state, policy, outcome) tuples where:
            - state: encoded board position
            - policy: move probability distribution used
            - outcome: final game result from the perspective of the player to move
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        net.to(device).eval()
        print("Model moved to device and set to eval mode")
    except Exception as e:
        print(f"Error setting up model: {e}")
        traceback.print_exc()
        return []
        
    examples = []

    for g in range(n_games):
        print(f"=== Starting game {g+1}/{n_games} ===")
        
        try:
            print("Creating board...")
            board = SimpleChessBoard()
            print("Board created and reset successfully")
            
            states, pis = [], []
            z = 0.0  # Default game outcome (draw)
            moves_played = 0
            
            # Play the game until completion or max moves
            while moves_played < max_plies:
                moves_played += 1
                print(f"Move {moves_played}")
                
                # Get legal moves
                legal_moves = board.get_legal_moves()
                
                if not legal_moves:
                    print("No legal moves - game over")
                    break
                
                print(f"Got {len(legal_moves)} legal moves")
                
                # Encode current board state
                state = encode_simple_board(board)
                state = state.to(device)
                
                # Get policy and value from neural network
                with torch.no_grad():
                    logits, value = net(state)
                
                # Select move using policy network
                uci, probs = _select_move_from_policy(logits, legal_moves, temperature)
                
                print(f"Selected move: {uci} (value: {value.item():.2f})")
                
                # Store policy for training
                pi = probs.cpu().detach().numpy().flatten()
                
                # Store state and policy
                states.append(state.cpu())
                pis.append(pi)
                
                # Apply move
                board.apply_move(uci)
                print(f"Applied move {uci}")
                
                # Check if game over
                done, result = board.is_game_over()
                if done:
                    print(f"Game over: {result}")
                    # Convert game result to numeric outcome
                    if result == "1-0":
                        z = 1.0
                    elif result == "0-1":
                        z = -1.0
                    else:
                        z = 0.0
                    break
            
            # Handle max plies reached
            if moves_played >= max_plies:
                print(f"Game {g+1}: reached max moves, ending as draw")
            
            # Save examples from this game
            print(f"Game {g+1} completed with {len(states)} positions")
            
            # Flip outcome for black's perspective on odd moves
            for i, (s, p) in enumerate(zip(states, pis)):
                example_z = z if (i % 2 == 0) else -z
                examples.append((s, p, example_z))
            
            # Save progress periodically
            if (g + 1) % 2 == 0:
                try:
                    torch.save(examples, "selfplay_partial.pt")
                    print(f"Saved {len(examples)} examples after {g+1} games")
                except Exception as e:
                    print(f"Error saving partial progress: {e}")
                    traceback.print_exc()
                
        except Exception as e:
            print(f"Error in game {g+1}: {e}")
            traceback.print_exc()

    # Save all examples at the end
    if examples:
        try:
            torch.save(examples, "selfplay.pt")
            print(f"Self-play complete: saved {len(examples)} examples")
        except Exception as e:
            print(f"Error saving final examples: {e}")
            traceback.print_exc()
    else:
        print("No examples generated during self-play")
    
    return examples


if __name__ == "__main__":
    # Load model if available
    model_path = "model_latest.pt" if os.path.exists("model_latest.pt") else None
    net = ChessNet()
    
    if model_path:
        try:
            print(f"Loading model from {model_path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model")
    else:
        print("No trained model found, using untrained model")
    
    # Run self-play with temperature control
    # Higher temperature = more exploration
    selfplay(net, n_games=10, max_plies=200, temperature=1.2)
```