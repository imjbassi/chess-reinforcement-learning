```python
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
    """Convert square index to UCI notation (e.g., 0 -> 'a1')"""
    rank, file = sq_to_coords(sq)
    return f"{chr(file + ord('a'))}{rank + 1}"


def move_to_uci(from_sq, to_sq):
    """Convert move to UCI notation (e.g., 'e2e4')"""
    return f"{sq_to_uci(from_sq)}{sq_to_uci(to_sq)}"


class SimpleChessBoard:
    """A simple chess board implementation with basic move generation and validation."""

    def __init__(self):
        self.board = None
        self.white_to_move = True
        self.moves_played = 0
        self.castling_rights = [True, True, True, True]
        self.en_passant_square = None
        self.reset()

    def reset(self):
        """Reset the board to the initial position."""
        self.board = INITIAL_BOARD.copy()
        self.white_to_move = True
        self.moves_played = 0
        self.castling_rights = [True, True, True, True]  # WK, WQ, BK, BQ
        self.en_passant_square = None

    def _is_attacked_on_board(self, board, sq, by_white):
        """Check if a square is attacked by the specified color on a given board state."""
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

        # Sliding pieces - Rook/Queen (orthogonal)
        for dr, df in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            r, f = rank, file
            while True:
                r += dr
                f += df
                if not (0 <= r < 8 and 0 <= f < 8):
                    break
                attacker = board[r, f]
                if attacker != 0:
                    if (by_white and attacker in (ROOK, QUEEN)) or \
                       (not by_white and attacker in (-ROOK, -QUEEN)):
                        return True
                    break

        # Sliding pieces - Bishop/Queen (diagonal)
        for dr, df in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            r, f = rank, file
            while True:
                r += dr
                f += df
                if not (0 <= r < 8 and 0 <= f < 8):
                    break
                attacker = board[r, f]
                if attacker != 0:
                    if (by_white and attacker in (BISHOP, QUEEN)) or \
                       (not by_white and attacker in (-BISHOP, -QUEEN)):
                        return True
                    break

        return False

    def _generate_pawn_moves(self, sq, rank, file, piece, is_white, moves):
        """Generate all pseudo-legal pawn moves from the given square."""
        direction = 1 if is_white else -1

        # Forward move
        new_rank = rank + direction
        if 0 <= new_rank < 8 and self.board[new_rank, file] == 0:
            # Check for promotion
            if (is_white and new_rank == 7) or (not is_white and new_rank == 0):
                for promo in ['q', 'r', 'b', 'n']:
                    moves.append(move_to_uci(sq, coords_to_sq(new_rank, file)) + promo)
            else:
                moves.append(move_to_uci(sq, coords_to_sq(new_rank, file)))

                # Double push from starting rank
                if (is_white and rank == 1) or (not is_white and rank == 6):
                    new_rank2 = rank + 2 * direction
                    if 0 <= new_rank2 < 8 and self.board[new_rank2, file] == 0:
                        moves.append(move_to_uci(sq, coords_to_sq(new_rank2, file)))

        # Captures
        for capture_file in [file - 1, file + 1]:
            if 0 <= capture_file < 8:
                new_rank = rank + direction
                if 0 <= new_rank < 8:
                    target_piece = self.board[new_rank, capture_file]
                    can_capture = (is_white and target_piece < 0) or (not is_white and target_piece > 0)
                    
                    # En passant capture
                    if not can_capture and self.en_passant_square == (new_rank, capture_file):
                        can_capture = True
                    
                    if can_capture:
                        # Check for promotion
                        if (is_white and new_rank == 7) or (not is_white and new_rank == 0):
                            for promo in ['q', 'r', 'b', 'n']:
                                moves.append(move_to_uci(sq, coords_to_sq(new_rank, capture_file)) + promo)
                        else:
                            moves.append(move_to_uci(sq, coords_to_sq(new_rank, capture_file)))

    def _generate_knight_moves(self, sq, rank, file, is_white, moves):
        """Generate all pseudo-legal knight moves from the given square."""
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for dr, df in knight_moves:
            new_rank, new_file = rank + dr, file + df
            if 0 <= new_rank < 8 and 0 <= new_file < 8:
                target = self.board[new_rank, new_file]
                if target == 0 or (is_white and target < 0) or (not is_white and target > 0):
                    moves.append(move_to_uci(sq, coords_to_sq(new_rank, new_file)))

    def _generate_king_moves(self, sq, rank, file, piece, is_white, moves):
        """Generate all legal king moves from the given square (already checks check)."""
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, df in king_moves:
            new_rank, new_file = rank + dr, file + df
            if 0 <= new_rank < 8 and 0 <= new_file < 8:
                target = self.board[new_rank, new_file]

                # Can't move onto own piece or capture any king
                if (is_white and target > 0) or (not is_white and target < 0):
                    continue
                if abs(target) == KING:
                    continue

                # Simulate move and check if king would be in check
                test_board = self.board.copy()
                test_board[rank, file] = 0
                test_board[new_rank, new_file] = piece

                new_king_sq = coords_to_sq(new_rank, new_file)

                if not self._is_attacked_on_board(test_board, new_king_sq, not is_white):
                    moves.append(move_to_uci(sq, coords_to_sq(new_rank, new_file)))

        # Castling
        if is_white and rank == 0:
            # Kingside castling
            if self.castling_rights[0] and self.board[0, 5] == 0 and self.board[0, 6] == 0:
                if not self._is_attacked_on_board(self.board, coords_to_sq(0, 4), False) and \
                   not self._is_attacked_on_board(self.board, coords_to_sq(0, 5), False) and \
                   not self._is_attacked_on_board(self.board, coords_to_sq(0, 6), False):
                    moves.append('e1g1')
            # Queenside castling
            if self.castling_rights[1] and self.board[0, 1] == 0 and self.board[0, 2] == 0 and self.board[0, 3] == 0:
                if not self._is_attacked_on_board(self.board, coords_to_sq(0, 4), False) and \
                   not self._is_attacked_on_board(self.board, coords_to_sq(0, 3), False) and \
                   not self._is_attacked_on_board(self.board, coords_to_sq(0, 2), False):
                    moves.append('e1c1')
        elif not is_white and rank == 7:
            # Kingside castling
            if self.castling_rights[2] and self.board[7, 5] == 0 and self.board[7, 6] == 0:
                if not self._is_attacked_on_board(self.board, coords_to_sq(7, 4), True) and \
                   not self._is_attacked_on_board(self.board, coords_to_sq(7, 5), True) and \
                   not self._is_attacked_on_board(self.board, coords_to_sq(7, 6), True):
                    moves.append('e8g8')
            # Queenside castling
            if self.castling_rights[3] and self.board[7, 1] == 0 and self.board[7, 2] == 0 and self.board[7, 3] == 0:
                if not self._is_attacked_on_board(self.board, coords_to_sq(7, 4), True) and \
                   not self._is_attacked_on_board(self.board, coords_to_sq(7, 3), True) and \
                   not self._is_attacked_on_board(self.board, coords_to_sq(7, 2), True):
                    moves.append('e8c8')

    def _generate_sliding_moves(self, sq, rank, file, piece_type, is_white, moves):
        """Generate all pseudo-legal sliding piece moves (rook, bishop, queen)."""
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

                # Can't move onto own piece or capture any king
                if (is_white and target > 0) or (not is_white and target < 0):
                    break
                if abs(target) == KING:
                    break

                moves.append(move_to_uci(sq, coords_to_sq(new_rank, new_file)))

                # Stop after capturing opponent piece
                if target != 0:
                    break

    def get_legal_moves(self):
        """Generate all legal moves, filtering out any that leave own king in check."""
        pseudo_moves = []

        # Generate pseudo-legal moves for all pieces
        for sq in range(64):
            rank, file = sq_to_coords(sq)
            piece = self.board[rank, file]

            # Skip empty squares and opponent pieces
            if piece == 0 or (piece > 0 and not self.white_to_move) or (piece < 0 and self.white_to_move):
                continue

            piece_type = abs(piece)
            is_white = piece > 0

            if piece_type == PAWN:
                self._generate_pawn_moves(sq, rank, file, piece, is_white, pseudo_moves)
            elif piece_type == KNIGHT:
                self._generate_knight_moves(sq, rank, file, is_white, pseudo_moves