import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Add current directory to path

import torch
import torch.nn.functional as F
import numpy as np
import traceback
import random
import threading
import time

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
    """Try to get legal moves with a timeout, fall back to Python if needed"""
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

def selfplay(net, n_games=10, max_plies=200, temperature=1.0):
    """Run self-play games using neural network policy for move selection"""
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
            while True:
                moves_played += 1
                print(f"Move {moves_played}")
                
                if moves_played > max_plies:
                    print(f"Game {g+1}: reached max moves, ending as draw")
                    break
                
                # Get legal moves
                legal = board.get_legal_moves()
                
                if not legal:
                    print("No legal moves - game over")
                    break
                
                print(f"Got {len(legal)} legal moves")
                
                # REPLACED: Random move selection with neural network policy
                state = encode_simple_board(board)
                state = state.to(device)
                
                with torch.no_grad():  # Disable gradients for inference
                    logits, value = net(state)
                
                # Create a mask for legal moves only
                mask = torch.full_like(logits, float('-inf'))
                for uci in legal:
                    f = ord(uci[0]) - ord('a') + (ord(uci[1]) - ord('1')) * 8
                    t = ord(uci[2]) - ord('a') + (ord(uci[3]) - ord('1')) * 8
                    idx = f * 64 + t
                    mask[0, idx] = logits[0, idx]
                
                # Apply temperature for exploration/exploitation control
                if temperature != 1.0:
                    mask = mask / temperature
                
                # Convert logits to probabilities
                probs = F.softmax(mask, dim=1)
                
                # Sample move from the probability distribution
                try:
                    move_idx = torch.multinomial(probs, num_samples=1).item()
                    
                    # Convert move index back to UCI
                    f, t = divmod(move_idx, 64)
                    uci = f"{chr(f % 8 + 97)}{f // 8 + 1}{chr(t % 8 + 97)}{t // 8 + 1}"
                    
                    # Verify move is legal
                    if uci not in legal:
                        raise ValueError(f"Selected move {uci} not in legal moves")
                        
                except (ValueError, RuntimeError) as e:
                    # Fallback to random move if sampling fails
                    print(f"Error sampling from policy: {e}")
                    uci = random.choice(legal)
                    print(f"Falling back to random move: {uci}")
                    
                    # Update probabilities for this move
                    probs = torch.zeros_like(probs)
                    f = ord(uci[0]) - ord('a') + (ord(uci[1]) - ord('1')) * 8
                    t = ord(uci[2]) - ord('a') + (ord(uci[3]) - ord('1')) * 8
                    idx = f * 64 + t
                    probs[0, idx] = 1.0
                
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
                    z = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0
                    break
            
            # Save examples from this game
            print(f"Game {g+1} completed with {len(states)} positions")
            
            for i, (s, p) in enumerate(zip(states, pis)):
                example_z = z if (i % 2 == 0) else -z
                examples.append((s, p, example_z))
            
            # Save progress periodically
            if (g+1) % 2 == 0:
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
    try:
        torch.save(examples, "selfplay.pt")
        print(f"Self-play complete: saved {len(examples)} examples")
    except Exception as e:
        print(f"Error saving final examples: {e}")
        traceback.print_exc()
    
    return examples

if __name__ == "__main__":
    # Load model if available
    model_path = "model_latest.pt" if os.path.exists("model_latest.pt") else None
    net = ChessNet()
    if model_path:
        try:
            print(f"Loading model from {model_path}")
            net.load_state_dict(torch.load(model_path, 
                                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model")
    else:
        print("No trained model found, using untrained model")
    
    # Run self-play with temperature control
    selfplay(net, n_games=10, max_plies=200, temperature=1.2)  # Higher temperature = more exploration
