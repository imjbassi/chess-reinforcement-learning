#!/usr/bin/env python3
import os
import sys
import pygame
import torch
import torch.nn.functional as F
import textwrap
import random

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import the C++ engine
try:
    # DISABLED: Force use of Python implementation only
    # from engine.chess_board import ChessBoard
    # from model.model import encode_board, ChessNet
    CPP_ENGINE_AVAILABLE = False
    print("C++ engine disabled - using Python-only implementation")
except Exception as e:
    print(f"Warning: C++ chess engine unavailable: {e}")
    print("Using Python-only chess implementation.")
    CPP_ENGINE_AVAILABLE = False

from model.model import ChessNet

SIZE      = 512
SQ        = SIZE // 8
COLORS    = [(235,235,208),(119,148,85)]

THIS_DIR  = os.path.dirname(__file__)
PROJECT   = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
ASSET_DIR = os.path.join(PROJECT, "assets")

PIECE_FILES = {
    'P':'wP.png','N':'wN.png','B':'wB.png','R':'wR.png',
    'Q':'wQ.png','K':'wK.png','p':'bP.png','n':'bN.png',
    'b':'bB.png','r':'bR.png','q':'bQ.png','k':'bK.png'
}

# Function to load chess piece images
def load_images():
    imgs = {}
    for k,fn in PIECE_FILES.items():
        surf = pygame.image.load(os.path.join(ASSET_DIR, fn)).convert_alpha()
        imgs[k] = pygame.transform.smoothscale(surf, (SQ, SQ))
    return imgs

# Function to draw the board and pieces using the C++ extension
def draw_board(scr, board, imgs):
    for r in range(8):
        for f in range(8):
            pygame.draw.rect(
                scr,
                COLORS[(r+f)%2],
                (f*SQ, (7-r)*SQ, SQ, SQ)
            )
    b = board._b
    order = ['P','N','B','R','Q','K','p','n','b','r','q','k']
    for idx,pc in enumerate(order):
        bb = b.pieces()[idx]
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            rr, ff = divmod(sq, 8)
            scr.blit(imgs[pc], (ff*SQ, (7-rr)*SQ))

# Function to draw arrows for moves
def draw_arrow(surface, start_sq, end_sq, color=(255,0,0), width=4):
    start_x = (start_sq % 8) * SQ + SQ // 2
    start_y = (7 - start_sq // 8) * SQ + SQ // 2
    end_x   = (end_sq % 8) * SQ + SQ // 2
    end_y   = (7 - end_sq // 8) * SQ + SQ // 2
    pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), width)
    dx, dy = end_x - start_x, end_y - start_y
    length = max((dx**2 + dy**2)**0.5, 1)
    ux, uy = dx / length, dy / length
    left = (-uy, ux)
    right = (uy, -ux)
    arrow_size = 12
    p1 = (end_x - arrow_size * (ux + 0.5 * left[0]), end_y - arrow_size * (uy + 0.5 * left[1]))
    p2 = (end_x - arrow_size * (ux + 0.5 * right[0]), end_y - arrow_size * (uy + 0.5 * right[1]))
    pygame.draw.polygon(surface, color, [(end_x, end_y), p1, p2])

# Function to check if a square is attacked - FIXED PAWN ATTACK DETECTION
def is_square_attacked(position, sq, by_white):
    """Check if a square is attacked by any piece of the given color."""
    # Handle None square
    if sq is None:
        return False
        
    # Check for pawn attacks - FIXED: Correct direction logic
    pawn_piece = 'P' if by_white else 'p'
    sq_rank, sq_file = divmod(sq, 8)
    
    # White pawns attack diagonally upward (from lower ranks to higher ranks)
    # Black pawns attack diagonally downward (from higher ranks to lower ranks)
    if by_white:
        # White pawns attack from rank below (rank - 1)
        attack_rank = sq_rank - 1
        if attack_rank >= 0:  # Valid rank
            # Check both diagonal attack squares
            for attack_file in [sq_file - 1, sq_file + 1]:
                if 0 <= attack_file < 8:  # Valid file
                    attack_sq = attack_rank * 8 + attack_file
                    if attack_sq in position and position[attack_sq] == pawn_piece:
                        return True
    else:
        # Black pawns attack from rank above (rank + 1)
        attack_rank = sq_rank + 1
        if attack_rank < 8:  # Valid rank
            # Check both diagonal attack squares
            for attack_file in [sq_file - 1, sq_file + 1]:
                if 0 <= attack_file < 8:  # Valid file
                    attack_sq = attack_rank * 8 + attack_file
                    if attack_sq in position and position[attack_sq] == pawn_piece:
                        return True
    
    # Check for knight attacks
    knight_piece = 'N' if by_white else 'n'
    knight_offsets = [6, 10, 15, 17, -6, -10, -15, -17]
    
    for offset in knight_offsets:
        attack_sq = sq - offset  # Reversed for attackers
        if 0 <= attack_sq < 64:
            # Check for valid knight move (not jumping across board edges)
            sq_rank, sq_file = divmod(sq, 8)
            att_rank, att_file = divmod(attack_sq, 8)
            if abs(sq_file - att_file) <= 2 and abs(sq_rank - att_rank) <= 2:
                if attack_sq in position and position[attack_sq] == knight_piece:
                    return True
    
    # Check for king attacks (for adjacent squares)
    king_piece = 'K' if by_white else 'k'
    king_offsets = [1, 7, 8, 9, -1, -7, -8, -9]
    
    for offset in king_offsets:
        attack_sq = sq - offset
        if 0 <= attack_sq < 64:
            # Check valid king move (not jumping across board edges)
            sq_rank, sq_file = divmod(sq, 8)
            att_rank, att_file = divmod(attack_sq, 8)
            if abs(sq_file - att_file) <= 1 and abs(sq_rank - att_rank) <= 1:
                if attack_sq in position and position[attack_sq] == king_piece:
                    return True # Found an attacking king
                    
    # FIXED: Sliding piece attacks (rook-like: horizontal/vertical)
    rook_pieces = ['R', 'Q'] if by_white else ['r', 'q']
    for direction in [1, -1, 8, -8]:  # right, left, down, up
        current_sq = sq
        sq_rank, sq_file = divmod(sq, 8)
        for i in range(1, 8):
            attack_sq = sq - i * direction  # Moving in reverse to find attacker
            
            if not (0 <= attack_sq < 64):
                break

            att_rank, att_file = divmod(attack_sq, 8)
            # Check if we've wrapped around the board
            if direction in [1, -1] and att_rank != sq_rank: # Horizontal
                break
            if direction in [8, -8] and att_file != sq_file: # Vertical
                break

            if attack_sq in position:
                if position[attack_sq] in rook_pieces:
                    return True
                break # Blocked by another piece
    
    # FIXED: Bishop-like attacks with similar corrections
    bishop_pieces = ['B', 'Q'] if by_white else ['b', 'q']
    for direction in [7, 9, -7, -9]:  # diagonals
        current_sq = sq
        sq_rank, sq_file = divmod(sq, 8)
        for i in range(1, 8):
            attack_sq = sq - i * direction
            
            if not (0 <= attack_sq < 64):
                break
            
            att_rank, att_file = divmod(attack_sq, 8)
            # Check for board wrap-around
            if abs(att_rank - sq_rank) != i or abs(att_file - sq_file) != i:
                break

            if attack_sq in position:
                if position[attack_sq] in bishop_pieces:
                    return True
                break # Blocked
            
    return False

# Function to run self-play and visualize the games - USING PYTHON CHESS ONLY
def selfplay_viz(n_games=2, delay=200):
    pygame.init()
    screen = pygame.display.set_mode((SIZE, SIZE + 180))
    pygame.display.set_caption("Chess‑RL Self‑Play Viz")
    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 24)
    imgs = load_images()
    
    # PYTHON-ONLY CHESS IMPLEMENTATION (No C++ extension)
    initial_position = {
        0: 'r', 1: 'n', 2: 'b', 3: 'q', 4: 'k', 5: 'b', 6: 'n', 7: 'r',
        8: 'p', 9: 'p', 10: 'p', 11: 'p', 12: 'p', 13: 'p', 14: 'p', 15: 'p',
        48: 'P', 49: 'P', 50: 'P', 51: 'P', 52: 'P', 53: 'P', 54: 'P', 55: 'P',
        56: 'R', 57: 'N', 58: 'B', 59: 'Q', 60: 'K', 61: 'B', 62: 'N', 63: 'R'
    }
    
    # Function to draw pieces from board state
    def draw_pieces(scr, state):
        # Draw board
        for r in range(8):
            for f in range(8):
                pygame.draw.rect(
                    scr,
                    COLORS[(r+f)%2],
                    (f*SQ, (7-r)*SQ, SQ, SQ)
                )
        
        # Draw pieces
        for sq, piece in state.items():
            r, f = divmod(sq, 8)
            scr.blit(imgs[piece], (f*SQ, (7-r)*SQ))
    
    print("Using Python-only chess implementation for selfplay visualization")
    
    try:
        # Try to load neural network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChessNet().to(device)
        model.eval()
        model_loaded = True
        print("Neural network loaded")
    except Exception as e:
        model_loaded = False
        print(f"Could not load neural network: {e}")
    
    screen.fill((0, 0, 0))
    screen.blit(font.render("Starting games...", True, (0, 200, 0)), (10, SIZE + 20))
    screen.blit(font.render(f"Using {'RL model' if model_loaded else 'random policy'}", True, (0, 200, 0)), (10, SIZE + 60))
    pygame.display.flip()
    pygame.time.wait(1000)
    
    print(f"DISABLED: C++ selfplay visualization - this function now uses Python chess only")

if __name__ == "__main__":
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((SIZE, SIZE + 180))
    pygame.display.set_caption("Chess‑RL Neural Network Viz")
    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 24)
    
    try:
        # Load chess pieces
        imgs = load_images()
        
        # Load neural network
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ChessNet().to(device)
            model.eval()
            model_loaded = True
            screen.fill((0, 0, 0))
            screen.blit(font.render("Neural network loaded successfully!", True, (0, 200, 0)), (10, SIZE + 20))
            pygame.display.flip()
            print("Neural network loaded")
            pygame.time.wait(1000)
        except Exception as e:
            model_loaded = False
            print(f"Could not load neural network: {e}")
            screen.fill((0, 0, 0))
            screen.blit(font.render(f"Neural network error: {str(e)[:40]}", True, (255, 0, 0)), (10, SIZE + 20))
            pygame.display.flip()
            pygame.time.wait(1000)
            
        # PYTHON-ONLY CHESS IMPLEMENTATION (No C++ extension)
        initial_position = {
            0: 'r', 1: 'n', 2: 'b', 3: 'q', 4: 'k', 5: 'b', 6: 'n', 7: 'r',
            8: 'p', 9: 'p', 10: 'p', 11: 'p', 12: 'p', 13: 'p', 14: 'p', 15: 'p',
            48: 'P', 49: 'P', 50: 'P', 51: 'P', 52: 'P', 53: 'P', 54: 'P', 55: 'P',
            56: 'R', 57: 'N', 58: 'B', 59: 'Q', 60: 'K', 61: 'B', 62: 'N', 63: 'R'
        }
        
        # Function to draw pieces from board state
        def draw_pieces(scr, state):
            # Draw board
            for r in range(8):
                for f in range(8):
                    pygame.draw.rect(
                        scr,
                        COLORS[(r+f)%2],
                        (f*SQ, (7-r)*SQ, SQ, SQ)
                    )
            
            # Draw pieces
            for sq, piece in state.items():
                r, f = divmod(sq, 8)
                scr.blit(imgs[piece], (f*SQ, (7-r)*SQ))
        
        # Generate simple legal moves for a piece (improved for better visualization)
        def get_simple_legal_moves(position, turn_white=True):
            moves = []

            # Find the king's square first
            king_piece = 'K' if turn_white else 'k'
            king_square = None
            for sq, piece in position.items():
                if piece == king_piece:
                    king_square = sq
                    break
            
            if king_square is None:
                # This should not happen in a legal game
                return []

            # Check if king is currently in check
            king_in_check = is_square_attacked(position, king_square, not turn_white)
            if king_in_check:
                print(f"DEBUG: {king_piece} is in check at square {king_square}")

            # Generate all pseudo-legal moves
            pseudo_moves = []
            for sq, piece in position.items():
                if (piece.isupper() and turn_white) or (piece.islower() and not turn_white):
                    r, f = divmod(sq, 8)
                    piece_type = piece.lower()
                    
                    # Pawn moves
                    if piece_type == 'p':
                        direction = -8 if turn_white else 8
                        start_rank = 6 if turn_white else 1
                        # one step forward
                        target = sq + direction
                        if 0 <= target < 64 and target not in position:
                            pseudo_moves.append((sq, target))
                            # two steps forward from starting position
                            if r == start_rank and (sq + 2*direction) not in position:
                                pseudo_moves.append((sq, sq + 2*direction))
                        # captures
                        for file_offset in [-1, 1]:
                            if 0 <= f + file_offset < 8:
                                target = sq + direction + file_offset
                                if 0 <= target < 64 and target in position and (position[target].isupper() != turn_white):
                                    pseudo_moves.append((sq, target))
                    
                    # Knight moves
                    elif piece_type == 'n':
                        knight_moves = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
                        for dr, df in knight_moves:
                            new_r, new_f = r + dr, f + df
                            if 0 <= new_r < 8 and 0 <= new_f < 8:
                                target = new_r * 8 + new_f
                                if target not in position or position[target].isupper() != turn_white:
                                    pseudo_moves.append((sq, target))
                    
                    # Bishop moves
                    elif piece_type == 'b':
                        for dr, df in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                            for i in range(1, 8):
                                new_r, new_f = r + i*dr, f + i*df
                                if 0 <= new_r < 8 and 0 <= new_f < 8:
                                    target = new_r * 8 + new_f
                                    if target not in position:
                                        pseudo_moves.append((sq, target))
                                    elif position[target].isupper() != turn_white:
                                        pseudo_moves.append((sq, target))
                                        break
                                    else:
                                        break
                                else:
                                    break
                    
                    # Rook moves
                    elif piece_type == 'r':
                        for dr, df in [(1,0), (-1,0), (0,1), (0,-1)]:
                            for i in range(1, 8):
                                new_r, new_f = r + i*dr, f + i*df
                                if 0 <= new_r < 8 and 0 <= new_f < 8:
                                    target = new_r * 8 + new_f
                                    if target not in position:
                                        pseudo_moves.append((sq, target))
                                    elif position[target].isupper() != turn_white:
                                        pseudo_moves.append((sq, target))
                                        break
                                    else:
                                        break
                                else:
                                    break
                    
                    # Queen moves (combination of rook and bishop)
                    elif piece_type == 'q':
                        for dr, df in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                            for i in range(1, 8):
                                new_r, new_f = r + i*dr, f + i*df
                                if 0 <= new_r < 8 and 0 <= new_f < 8:
                                    target = new_r * 8 + new_f
                                    if target not in position:
                                        pseudo_moves.append((sq, target))
                                    elif position[target].isupper() != turn_white:
                                        pseudo_moves.append((sq, target))
                                        break
                                    else:
                                        break
                                else:
                                    break
                    
                    # King moves
                    elif piece_type == 'k':
                        for dr, df in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                            new_r, new_f = r + dr, f + df
                            if 0 <= new_r < 8 and 0 <= new_f < 8:
                                target = new_r * 8 + new_f
                                if target not in position or position[target].isupper() != turn_white:
                                    pseudo_moves.append((sq, target))

            print(f"DEBUG: Generated {len(pseudo_moves)} pseudo-legal moves for {'White' if turn_white else 'Black'}")

            # Filter out moves that leave the king in check
            legal_count = 0
            for orig_sq, dest_sq in pseudo_moves:
                test_position = position.copy()
                piece = test_position.pop(orig_sq)
                
                # Handle captures - remove the captured piece
                if dest_sq in test_position:
                    test_position.pop(dest_sq)
                
                # Place the piece at its new position
                test_position[dest_sq] = piece
                
                # Determine where the king will be after this move
                current_king_square = king_square
                if piece.lower() == 'k':
                    current_king_square = dest_sq

                # Only include moves that don't leave the king in check
                if not is_square_attacked(test_position, current_king_square, not turn_white):
                    moves.append(sq_to_uci(orig_sq) + sq_to_uci(dest_sq))
                    legal_count += 1
                else:
                    if piece.lower() == 'k':
                        print(f"DEBUG: King move {sq_to_uci(orig_sq)}{sq_to_uci(dest_sq)} rejected - would be in check")

            print(f"DEBUG: {legal_count} legal moves found for {'White' if turn_white else 'Black'}")
            if king_in_check and legal_count == 0:
                print(f"DEBUG: CHECKMATE - {king_piece} has no legal moves!")
            elif king_in_check:
                print(f"DEBUG: King in check, found {legal_count} moves to escape")
            
            return moves
        
        # Convert square index to UCI notation
        def sq_to_uci(sq):
            rank, file = divmod(sq, 8)
            return chr(97 + file) + str(rank + 1)
        
        # Apply a move to the position (fixed version to prevent piece transformations)
        def apply_move(position, move):
            position = position.copy()  # Create a copy to avoid modifying the original
            
            from_file = ord(move[0]) - ord('a')
            from_rank = int(move[1]) - 1
            to_file = ord(move[2]) - ord('a')
            to_rank = int(move[3]) - 1
            
            from_sq = from_rank * 8 + from_file
            to_sq = to_rank * 8 + to_file
            
            # First check if there's a piece to move
            if from_sq in position:
                piece = position.pop(from_sq)  # Remove from source
                
                # Check if there's a capture (remove captured piece first)
                if to_sq in position:
                    position.pop(to_sq)
                    
                # Place the moved piece
                position[to_sq] = piece
                
                # Handle castling (king move of 2 squares)
                if piece in ['K', 'k'] and abs(from_file - to_file) > 1:
                    # Kingside castling (king moved right 2 squares)
                    if to_file > from_file:
                        # Move the rook from h-file to f-file
                        rook_from = from_rank * 8 + 7  # h-file
                        rook_to = from_rank * 8 + 5    # f-file
                        if rook_from in position and position[rook_from] in ['R', 'r']:
                            rook = position.pop(rook_from)
                            position[rook_to] = rook
                    # Queenside castling (king moved left 2 squares)
                    else:
                        # Move the rook from a-file to d-file
                        rook_from = from_rank * 8 + 0  # a-file
                        rook_to = from_rank * 8 + 3    # d-file
                        if rook_from in position and position[rook_from] in ['R', 'r']:
                            rook = position.pop(rook_from)
                            position[rook_to] = rook
                    
                # Handle pawn promotion (simplified - always promote to queen)
                if piece == 'P' and to_rank == 0:  # White pawn on 8th rank
                    position[to_sq] = 'Q'
                elif piece == 'p' and to_rank == 7:  # Black pawn on 1st rank
                    position[to_sq] = 'q'
            
            return position
            
        # Helper function to convert UCI move string to square indices
        def move_to_squares(move):
            from_file = ord(move[0]) - ord('a')
            from_rank = int(move[1]) - 1
            to_file = ord(move[2]) - ord('a')
            to_rank = int(move[3]) - 1
            from_sq = from_rank * 8 + from_file
            to_sq = to_rank * 8 + to_file
            return from_sq, to_sq
            
        # Create tensor representation of board (properly matching model's expected 18 channels)
        def encode_position_simple(position, white_to_move=True):
            # 18 channels to match model's expected input
            tensor = torch.zeros((1, 18, 8, 8), dtype=torch.float32)
            
            # Map pieces to planes (similar to the real encode_board function)
            piece_to_plane = {
                'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
            }
            
            # Fill tensor with piece positions
            for sq, piece in position.items():
                if piece in piece_to_plane:
                    plane = piece_to_plane[piece]
                    rank, file = divmod(sq, 8)
                    tensor[0, plane, rank, file] = 1.0
            
            # Set turn plane (all 1s for current player to move)
            tensor[0, 12, :, :] = 1.0 if white_to_move else 0.0
            
            # Fill remaining planes with features the model might expect
            # Plane 13: En passant possibilities
            # Plane 14: Castling rights
            # Plane 15-17: Other game state features
            
            return tensor
            
        # Game termination check function
        def is_game_over(position, white_to_move):
            # Check for checkmate or stalemate (no legal moves)
            moves = get_simple_legal_moves(position, white_to_move)
            if not moves:
                # Check if king is in check to determine if it's checkmate or stalemate
                king_piece = 'K' if white_to_move else 'k'
                king_square = None
                for sq, piece in position.items():
                    if piece == king_piece:
                        king_square = sq
                        break
                        
                if king_square and is_square_attacked(position, king_square, not white_to_move):
                    return True, "Checkmate" + (" - Black wins!" if white_to_move else " - White wins!")
                else:
                    return True, "Stalemate - Draw!"
                    
            # Check for insufficient material (kings only, or king+bishop/knight vs king)
            piece_count = {'K': 0, 'Q': 0, 'R': 0, 'B': 0, 'N': 0, 'P': 0, 
                           'k': 0, 'q': 0, 'r': 0, 'b': 0, 'n': 0, 'p': 0}
            
            for piece in position.values():
                piece_count[piece] += 1
            
            # Kings only
            if sum(piece_count.values()) == 2:
                return True, "Insufficient material - Draw!"
                
            # King vs king + knight/bishop
            if sum(piece_count.values()) == 3:
                if piece_count['B'] == 1 or piece_count['N'] == 1 or piece_count['b'] == 1 or piece_count['n'] == 1:
                    return True, "Insufficient material - Draw!"
                    
            return False, ""

        # Run multiple self-play games
        for game_num in range(1, 3):
            # Reset position for new game
            position = initial_position.copy()
            
            # Show starting position
            screen.fill((0, 0, 0))
            draw_pieces(screen, position)
            screen.blit(font.render(f"Game {game_num} - Starting Position", True, (0, 200, 0)), (10, SIZE + 20))
            screen.blit(small_font.render("Neural Network Visualization", True, (200, 200, 0)), (10, SIZE + 60))
            pygame.display.flip()
            pygame.time.wait(1000)
            
            # Game loop
            moves = []
            white_to_move = True
            game_over = False
            result = ""
            max_moves = 150  # Higher safety limit for complete games
            
            # Continue playing until game is over
            while not game_over and len(moves) < max_moves:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                
                # Generate legal moves using our simple but improved function
                legal_moves = get_simple_legal_moves(position, white_to_move)
                
                # Check for game over conditions
                game_over, result = is_game_over(position, white_to_move)
                if game_over:
                    break
                    
                # Select move using neural network if available
                if model_loaded:
                    # Get model predictions and select move (fixed variable scope issue)
                    try:
                        # Get board representation
                        state = encode_position_simple(position, white_to_move).to(device)
                        
                        # Get model predictions
                        with torch.no_grad():
                            logits, _ = model(state)
                        
                        # Create mask for legal moves
                        mask = torch.full_like(logits, float('-inf'))
                        for move in legal_moves:
                            from_file = ord(move[0]) - ord('a')
                            from_rank = int(move[1]) - 1
                            to_file = ord(move[2]) - ord('a')
                            to_rank = int(move[3]) - 1
                            from_sq = from_rank * 8 + from_file
                            to_sq = to_rank * 8 + to_file
                            idx = from_sq * 64 + to_sq
                            if idx < mask.shape[1]:
                                mask[0, idx] = logits[0, idx]
                        
                        # Get probabilities and sample
                        probs = F.softmax(mask, dim=1)
                        action = torch.multinomial(probs, 1).item()
                        from_sq, to_sq = divmod(action, 64)
                        
                        # Convert to UCI format
                        from_file = chr(97 + from_sq % 8)
                        from_rank = str(from_sq // 8 + 1)
                        to_file = chr(97 + to_sq % 8)
                        to_rank = str(to_sq // 8 + 1)
                        move = from_file + from_rank + to_file + to_rank
                        
                        # FIXED: Initialize legal_move_found variable before using it
                        legal_move_found = (move in legal_moves)
                        
                        # Fall back to a legal move if necessary
                        if not legal_move_found:
                            for legal_move in legal_moves:
                                if legal_move.startswith(move[:2]):  # Same starting position
                                    move = legal_move
                                    legal_move_found = True
                                    break
                        
                        # If still not found, use random
                        if not legal_move_found:
                            move = random.choice(legal_moves)  # Last resort fallback
                        
                        # Generate candidate moves
                        candidates = []
                        values, indices = torch.topk(probs, 6, dim=1)
                        for i in range(1, min(6, len(indices[0]))):
                            idx = indices[0][i].item()
                            from_sq, to_sq = divmod(idx, 64)
                            from_file = chr(97 + from_sq % 8)
                            from_rank = str(from_sq // 8 + 1)
                            to_file = chr(97 + to_sq % 8)
                            to_rank = str(to_sq // 8 + 1)
                            cand_move = from_file + from_rank + to_file + to_rank
                            if cand_move in legal_moves:
                                candidates.append(cand_move)
                    except Exception as e:
                        print(f"Model error: {e}")
                        move = random.choice(legal_moves)
                        candidates = random.sample(legal_moves, min(5, len(legal_moves))) if len(legal_moves) > 1 else []
                        values = torch.tensor([[0.8, 0.1, 0.05, 0.03, 0.02, 0.01]])  # Dummy probabilities
                else:
                    # Random selection
                    move = random.choice(legal_moves)
                    remaining = [m for m in legal_moves if m != move]
                    candidates = random.sample(remaining, min(5, len(remaining))) if remaining else []
                    values = torch.tensor([[0.8, 0.1, 0.05, 0.03, 0.02, 0.01]])  # Dummy probabilities
        
                # Draw current board with arrows
                screen.fill((0, 0, 0))
                draw_pieces(screen, position)
                
                # Draw arrows for candidate moves
                for j, candidate in enumerate(candidates):
                    if j >= 5:  # Limit to 5 candidate moves
                        break
                    from_sq, to_sq = move_to_squares(candidate)
                    colors = [(100, 255, 100), (100, 200, 255), (255, 255, 100), (255, 100, 255), (100, 255, 255)]
                    draw_arrow(screen, from_sq, to_sq, color=colors[j % len(colors)], width=2)
                
                # Draw arrow for current move
                from_sq, to_sq = move_to_squares(move)
                draw_arrow(screen, from_sq, to_sq, color=(255, 50, 50), width=4)
                
                # Display game info
                turn = "White" if white_to_move else "Black"
                move_number = (len(moves) // 2) + 1
                
                screen.blit(font.render(f"Game {game_num}", True, (0, 200, 0)), (10, SIZE + 20))
                screen.blit(small_font.render(f"Move {move_number}: {turn} plays {move}", True, (200, 200, 0)), 
                           (10, SIZE + 60))
                screen.blit(small_font.render(f"Move {len(moves)} | Showing {len(candidates)} candidate moves", 
                                          True, (180, 180, 180)), (10, SIZE + 100))
                
                # Display probabilities
                if model_loaded:
                    try:
                        # Get and display top move probabilities
                        top_values = values[0][:6].cpu().numpy()
                        
                        # Draw probability bars
                        for j, prob in enumerate(top_values[1:6]):
                            if j >= 5:
                                break
                            bar_width = int(150 * prob)
                            colors = [(100, 255, 100), (100, 200, 255), (255, 255, 100), (255, 100, 255), (100, 255, 255)]
                            pygame.draw.rect(
                                screen, 
                                colors[j % len(colors)],
                                (SIZE - 180, SIZE + 20 + j*20, bar_width, 15)
                            )
                            screen.blit(small_font.render(f"{prob:.3f}", True, (255, 255, 255)), 
                                      (SIZE - 180 + bar_width + 5, SIZE + 20 + j*20))
                    except Exception as e:
                        print(f"Error displaying probabilities: {e}")
                
                pygame.display.flip()
                pygame.time.wait(200)  # Animation speed
                
                # Apply move
                position = apply_move(position, move)
                moves.append(move)
                white_to_move = not white_to_move  # Switch turns
            
            # End of game
            screen.blit(font.render(f"Game completed: {result}", True, (255, 0, 0)), (10, SIZE + 140))
            pygame.display.flip()
            pygame.time.wait(2000)
        
        # Show completion
        screen.fill((0, 0, 0))
        draw_pieces(screen, position)
        screen.blit(font.render("All games complete!", True, (0, 255, 0)), (10, SIZE + 20))
        screen.blit(font.render("Close window to exit", True, (255, 255, 255)), (10, SIZE + 60))
        pygame.display.flip()
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Show error on screen
        screen.fill((0, 0, 0))
        error_msg = f"Critical error: {str(e)}"
        lines = textwrap.wrap(error_msg, 40)
        y = 40
        for line in lines:
            screen.blit(font.render(line, True, (255, 0, 0)), (20, y))
            y += 40
        pygame.display.flip()
    
    # Keep window open until closed
    print("Visualization complete. Close window to exit.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.time.wait(100)
    
    pygame.quit()

def is_move_safe_for_king(position, move, turn_white):
    """Verify a move doesn't leave the king in check."""
    test_position = apply_move(position.copy(), move)
    
    # Find king's position after move
    king_piece = 'K' if turn_white else 'k'
    king_sq = None
    for sq, piece in test_position.items():
        if piece == king_piece:
            king_sq = sq
            break
    
    # Check if king is in check after move
    if king_sq is None or is_square_attacked(test_position, king_sq, not turn_white):
        return False
    return True
