"""Simple Python chess implementation for fallback when C++ engine fails"""
import numpy as np
import torch

# Piece encodings
EMPTY = 0
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(1, 7)
WHITE, BLACK = 0, 1

# Board representation for a new game
INITIAL_BOARD = np.zeros((8, 8), dtype=int)
# Place pawns
INITIAL_BOARD[1, :] = PAWN
INITIAL_BOARD[6, :] = -PAWN
# Place pieces
INITIAL_BOARD[0, [0, 7]] = ROOK
INITIAL_BOARD[7, [0, 7]] = -ROOK
INITIAL_BOARD[0, [1, 6]] = KNIGHT
INITIAL_BOARD[7, [1, 6]] = -KNIGHT
INITIAL_BOARD[0, [2, 5]] = BISHOP
INITIAL_BOARD[7, [2, 5]] = -BISHOP
INITIAL_BOARD[0, 3] = QUEEN
INITIAL_BOARD[7, 3] = -QUEEN
INITIAL_BOARD[0, 4] = KING
INITIAL_BOARD[7, 4] = -KING

def sq_to_coords(sq):
    """Convert 0-63 square index to (rank, file)"""
    return divmod(sq, 8)

def coords_to_sq(rank, file):
    """Convert (rank, file) to 0-63 square index"""
    return rank * 8 + file

def sq_to_uci(sq):
    """Convert square index to UCI notation"""
    rank, file = sq_to_coords(sq)
    return f"{chr(file + ord('a'))}{rank + 1}"

def move_to_uci(from_sq, to_sq):
    """Convert move to UCI notation"""
    return f"{sq_to_uci(from_sq)}{sq_to_uci(to_sq)}"

class SimpleChessBoard:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = INITIAL_BOARD.copy()
        self.white_to_move = True
        self.moves_played = 0
        self.castling_rights = [True, True, True, True]  # WK, WQ, BK, BQ
        self.en_passant_square = None
    
    def get_legal_moves(self):
        """Generate all legal moves, enforcing king safety and piece rules."""
        moves = []
        # Find own king square
        king_sq = None
        for sq in range(64):
            rank, file = sq_to_coords(sq)
            piece = self.board[rank, file]
            if (self.white_to_move and piece == KING) or (not self.white_to_move and piece == -KING):
                king_sq = sq
                break
        
        # Helper: is a square attacked by opponent on a given board?
        def is_attacked_on_board(board, sq, by_white):
            rank, file = sq_to_coords(sq)
            # Pawn attacks
            direction = 1 if by_white else -1
            for df in [-1, 1]:
                r, f = rank - direction, file + df
                if 0 <= r < 8 and 0 <= f < 8:
                    attacker = board[r, f]
                    if (by_white and attacker == PAWN) or (not by_white and attacker == -PAWN):
                        return True
            # Knight attacks
            knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            for dr, df in knight_moves:
                r, f = rank + dr, file + df
                if 0 <= r < 8 and 0 <= f < 8:
                    attacker = board[r, f]
                    if (by_white and attacker == KNIGHT) or (not by_white and attacker == -KNIGHT):
                        return True
            # King attacks (adjacent enemy king)
            king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for dr, df in king_moves:
                r, f = rank + dr, file + df
                if 0 <= r < 8 and 0 <= f < 8:
                    attacker = board[r, f]
                    if (by_white and attacker == KING) or (not by_white and attacker == -KING):
                        return True
            # Sliding pieces
            # Rook/Queen
            for dr, df in [(0,1),(1,0),(0,-1),(-1,0)]:
                r, f = rank, file
                while True:
                    r += dr
                    f += df
                    if not (0 <= r < 8 and 0 <= f < 8):
                        break
                    attacker = board[r, f]
                    if attacker != 0:
                        if (by_white and (attacker == ROOK or attacker == QUEEN)) or (not by_white and (attacker == -ROOK or attacker == -QUEEN)):
                            return True
                        break
            # Bishop/Queen
            for dr, df in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                r, f = rank, file
                while True:
                    r += dr
                    f += df
                    if not (0 <= r < 8 and 0 <= f < 8):
                        break
                    attacker = board[r, f]
                    if attacker != 0:
                        if (by_white and (attacker == BISHOP or attacker == QUEEN)) or (not by_white and (attacker == -BISHOP or attacker == -QUEEN)):
                            return True
                        break
            return False
        # Generate pseudo-legal moves
        for sq in range(64):
            rank, file = sq_to_coords(sq)
            piece = self.board[rank, file]
            if (piece == 0) or (piece > 0 and not self.white_to_move) or (piece < 0 and self.white_to_move):
                continue
            piece_type = abs(piece)
            is_white = piece > 0
            # Pawn moves
            if piece_type == PAWN:
                direction = 1 if is_white else -1
                # Forward move
                new_rank = rank + direction
                if 0 <= new_rank < 8 and self.board[new_rank, file] == 0:
                    # Promotion check
                    if (is_white and new_rank == 7) or (not is_white and new_rank == 0):
                        moves.append(move_to_uci(sq, coords_to_sq(new_rank, file)) + 'q')
                    else:
                        moves.append(move_to_uci(sq, coords_to_sq(new_rank, file)))
                    # Double push from starting rank
                    if (is_white and rank == 1) or (not is_white and rank == 6):
                        new_rank2 = rank + 2 * direction
                        if 0 <= new_rank2 < 8 and self.board[new_rank2, file] == 0 and self.board[new_rank, file] == 0:
                            moves.append(move_to_uci(sq, coords_to_sq(new_rank2, file)))
                # Captures
                for capture_file in [file-1, file+1]:
                    if 0 <= capture_file < 8:
                        new_rank = rank + direction
                        if 0 <= new_rank < 8:
                            target_piece = self.board[new_rank, capture_file]
                            if (is_white and target_piece < 0) or (not is_white and target_piece > 0):
                                # Promotion check
                                if (is_white and new_rank == 7) or (not is_white and new_rank == 0):
                                    moves.append(move_to_uci(sq, coords_to_sq(new_rank, capture_file)) + 'q')
                                else:
                                    moves.append(move_to_uci(sq, coords_to_sq(new_rank, capture_file)))
            # Knight moves
            elif piece_type == KNIGHT:
                knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
                for dr, df in knight_moves:
                    new_rank, new_file = rank + dr, file + df
                    if 0 <= new_rank < 8 and 0 <= new_file < 8:
                        target = self.board[new_rank, new_file]
                        if target == 0 or (is_white and target < 0) or (not is_white and target > 0):
                            moves.append(move_to_uci(sq, coords_to_sq(new_rank, new_file)))
            # King moves
            elif piece_type == KING:
                king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                for dr, df in king_moves:
                    new_rank, new_file = rank + dr, file + df
                    if 0 <= new_rank < 8 and 0 <= new_file < 8:
                        target = self.board[new_rank, new_file]
                        # Can't move onto own piece
                        if (is_white and target > 0) or (not is_white and target < 0):
                            continue
                        # Can't capture own king
                        if (is_white and target == KING) or (not is_white and target == -KING):
                            continue
                        # Can't capture opponent king
                        if (is_white and target == -KING) or (not is_white and target == KING):
                            continue
                        
                        # Simulate move and check if king would be attacked
                        test_board = self.board.copy()
                        test_board[rank, file] = 0
                        test_board[new_rank, new_file] = piece
                        
                        # The new king square after this move
                        new_king_sq = coords_to_sq(new_rank, new_file)
                        
                        # Check if king would be attacked after this move
                        if not is_attacked_on_board(test_board, new_king_sq, not is_white):
                            moves.append(move_to_uci(sq, coords_to_sq(new_rank, new_file)))
            # Rook, Bishop, Queen moves
            elif piece_type in [ROOK, BISHOP, QUEEN]:
                directions = []
                if piece_type in [ROOK, QUEEN]:
                    directions.extend([(0, 1), (1, 0), (0, -1), (-1, 0)])
                if piece_type in [BISHOP, QUEEN]:
                    directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
                for dr, df in directions:
                    new_rank, new_file = rank, file
                    while True:
                        new_rank += dr
                        new_file += df
                        if not (0 <= new_rank < 8 and 0 <= new_file < 8):
                            break
                        target = self.board[new_rank, new_file]
                        # Can't move onto own piece or own king
                        if (is_white and target > 0) or (not is_white and target < 0):
                            break
                        if (is_white and target == KING) or (not is_white and target == -KING):
                            break
                        moves.append(move_to_uci(sq, coords_to_sq(new_rank, new_file)))
                        if target != 0:
                            break
        # Final filter: remove any moves that land on own king or capture opponent king
        filtered = []
        for m in moves:
            # Parse move
            from_file = ord(m[0]) - ord('a')
            from_rank = int(m[1]) - 1
            to_file = ord(m[2]) - ord('a')
            to_rank = int(m[3]) - 1
            target = self.board[to_rank, to_file]
            # Remove moves that land on own king or capture opponent king
            if (self.white_to_move and target == KING) or (not self.white_to_move and target == -KING):
                continue
            filtered.append(m)
        return filtered
    
    def apply_move(self, uci):
        """Apply a move in UCI format with proper tracking of special states"""
        from_file = ord(uci[0]) - ord('a')
        from_rank = int(uci[1]) - 1
        to_file = ord(uci[2]) - ord('a')
        to_rank = int(uci[3]) - 1
        
        piece = self.board[from_rank, from_file]
        piece_type = abs(piece)
        is_white = piece > 0
        
        # Check for castling rights updates
        if piece_type == KING:
            # King move means lose all castling rights for that color
            if is_white:
                self.castling_rights[0] = False  # WK
                self.castling_rights[1] = False  # WQ
            else:
                self.castling_rights[2] = False  # BK
                self.castling_rights[3] = False  # BQ
        elif piece_type == ROOK:
            # Rook move might affect castling rights
            if is_white:
                if from_rank == 0 and from_file == 0:
                    self.castling_rights[1] = False  # WQ
                elif from_rank == 0 and from_file == 7:
                    self.castling_rights[0] = False  # WK
            else:
                if from_rank == 7 and from_file == 0:
                    self.castling_rights[3] = False  # BQ
                elif from_rank == 7 and from_file == 7:
                    self.castling_rights[2] = False  # BK
    
        # Check for en passant
        self.en_passant_square = None
        if piece_type == PAWN:
            # Double push from starting rank
            if abs(from_rank - to_rank) == 2:
                mid_rank = (from_rank + to_rank) // 2
                self.en_passant_square = (mid_rank, from_file)
    
        # Move the piece
        self.board[from_rank, from_file] = 0
        self.board[to_rank, to_file] = piece
    
        # Toggle whose turn it is
        self.white_to_move = not self.white_to_move
        self.moves_played += 1
    
    def is_game_over(self):
        """Check if the game is over"""
        # Simplified game-over detection
        # 1. No pieces left of one color
        white_pieces = (self.board > 0).any()
        black_pieces = (self.board < 0).any()
        
        if not white_pieces:
            return True, "0-1"
        if not black_pieces:
            return True, "1-0"
        
        # 2. Move limit reached
        if self.moves_played >= 100:
            return True, "1/2-1/2"
        
        # 3. No legal moves (simplified)
        if not self.get_legal_moves():
            # Stalemate (no check detection in this simple version)
            return True, "1/2-1/2"
        
        return False, ""

def encode_simple_board(board):
    """Encode board for the neural network - with 18 channels to match C++ version"""
    # 18 planes total:
    # - 12 for pieces (6 white, 6 black)
    # - 1 for side to move
    # - 4 for castling rights
    # - 1 for en passant possibility
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    
    # Piece planes (same as before)
    piece_map = {
        PAWN: 0, KNIGHT: 1, BISHOP: 2, ROOK: 3, QUEEN: 4, KING: 5,
        -PAWN: 6, -KNIGHT: 7, -BISHOP: 8, -ROOK: 9, -QUEEN: 10, -KING: 11
    }
    
    for rank in range(8):
        for file in range(8):
            piece = board.board[rank, file]
            if piece != 0:
                planes[piece_map[piece], rank, file] = 1
    
    # Side to move (plane 12)
    if board.white_to_move:
        planes[12, :, :] = 1
    
    # Castling rights (planes 13-16)
    # WK, WQ, BK, BQ
    for i, has_right in enumerate(board.castling_rights):
        if has_right:
            planes[13 + i, :, :] = 1
    
    # En passant possibility (plane 17)
    if board.en_passant_square is not None:
        rank, file = board.en_passant_square
        planes[17, rank, file] = 1
    
    return torch.tensor(planes).unsqueeze(0)  # Add batch dimension