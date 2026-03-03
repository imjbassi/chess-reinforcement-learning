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
                    if (by_white and (attacker == ROOK or attacker == QUEEN)) or \
                       (not by_white and (attacker == -ROOK or attacker == -QUEEN)):
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
                    if (by_white and (attacker == BISHOP or attacker == QUEEN)) or \
                       (not by_white and (attacker == -BISHOP or attacker == -QUEEN)):
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
                moves.append(move_to_uci(sq, coords_to_sq(new_rank, file)) + 'q')
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
                    if (is_white and target_piece < 0) or (not is_white and target_piece > 0):
                        # Check for promotion
                        if (is_white and new_rank == 7) or (not is_white and new_rank == 0):
                            moves.append(move_to_uci(sq, coords_to_sq(new_rank, capture_file)) + 'q')
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
                self._generate_knight_moves(sq, rank, file, is_white, pseudo_moves)
            elif piece_type == KING:
                # King moves already check for check in _generate_king_moves
                self._generate_king_moves(sq, rank, file, piece, is_white, pseudo_moves)
            elif piece_type in [ROOK, BISHOP, QUEEN]:
                self._generate_sliding_moves(sq, rank, file, piece_type, is_white, pseudo_moves)

        # Filter moves that leave own king in check
        king_val = KING if self.white_to_move else -KING
        promo_map = {'q': QUEEN, 'r': ROOK, 'b': BISHOP, 'n': KNIGHT}
        legal = []

        for m in pseudo_moves:
            mf = ord(m[0]) - ord('a')
            mr = int(m[1]) - 1
            tf = ord(m[2]) - ord('a')
            tr = int(m[3]) - 1

            test_board = self.board.copy()
            moved_piece = test_board[mr, mf]
            test_board[mr, mf] = 0

            if len(m) > 4:
                promo_type = promo_map.get(m[4], QUEEN)
                test_board[tr, tf] = promo_type if self.white_to_move else -promo_type
            else:
                test_board[tr, tf] = moved_piece

            # Find king position after this move
            king_sq = None
            for s in range(64):
                r, f = sq_to_coords(s)
                if test_board[r, f] == king_val:
                    king_sq = s
                    break

            if king_sq is not None and not self._is_attacked_on_board(test_board, king_sq, not self.white_to_move):
                legal.append(m)

        return legal

    def apply_move(self, uci):
        """Apply a move in UCI format with proper tracking of special states."""
        from_file = ord(uci[0]) - ord('a')
        from_rank = int(uci[1]) - 1
        to_file = ord(uci[2]) - ord('a')
        to_rank = int(uci[3]) - 1

        piece = self.board[from_rank, from_file]
        piece_type = abs(piece)
        is_white = piece > 0

        # Update castling rights
        if piece_type == KING:
            if is_white:
                self.castling_rights[0] = False  # WK
                self.castling_rights[1] = False  # WQ
            else:
                self.castling_rights[2] = False  # BK
                self.castling_rights[3] = False  # BQ
        elif piece_type == ROOK:
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

        # Update en passant square
        self.en_passant_square = None
        if piece_type == PAWN and abs(from_rank - to_rank) == 2:
            mid_rank = (from_rank + to_rank) // 2
            self.en_passant_square = (mid_rank, from_file)

        # Handle pawn promotion
        if len(uci) > 4:
            promotion_piece = uci[4].lower()
            promotion_map = {'q': QUEEN, 'r': ROOK, 'b': BISHOP, 'n': KNIGHT}
            new_piece_type = promotion_map.get(promotion_piece, QUEEN)
            self.board[from_rank, from_file] = 0
            self.board[to_rank, to_file] = new_piece_type if is_white else -new_piece_type
        else:
            # Move the piece
            self.board[from_rank, from_file] = 0
            self.board[to_rank, to_file] = piece

            # Handle castling (king moves 2 squares)
            if piece_type == KING and abs(from_file - to_file) == 2:
                if to_file > from_file:  # Kingside
                    rook_from_file, rook_to_file = 7, 5
                else:  # Queenside
                    rook_from_file, rook_to_file = 0, 3
                rook = self.board[from_rank, rook_from_file]
                self.board[from_rank, rook_from_file] = 0
                self.board[from_rank, rook_to_file] = rook

        self.white_to_move = not self.white_to_move
        self.moves_played += 1

    def is_game_over(self):
        """Check if the game is over.

        Returns:
            tuple: (done, result) where result is '1-0', '0-1', or '1/2-1/2'.
        """
        legal = self.get_legal_moves()

        if not legal:
            # Find the current player's king
            king_val = KING if self.white_to_move else -KING
            for sq in range(64):
                rank, file = sq_to_coords(sq)
                if self.board[rank, file] == king_val:
                    if self._is_attacked_on_board(self.board, sq, not self.white_to_move):
                        # Checkmate: current player lost
                        return True, "0-1" if self.white_to_move else "1-0"
                    else:
                        return True, "1/2-1/2"
            return True, "1/2-1/2"

        # Draw by move limit
        if self.moves_played >= 200:
            return True, "1/2-1/2"

        return False, ""


def encode_simple_board(board):
    """Encode a SimpleChessBoard as a tensor for neural network input.

    Returns:
        torch.Tensor: Shape (1, 18, 8, 8) matching ChessNet's expected input.
    """
    piece_planes = {PAWN: 0, KNIGHT: 1, BISHOP: 2, ROOK: 3, QUEEN: 4, KING: 5}
    planes = [np.zeros((8, 8), dtype=np.float32) for _ in range(12)]

    for r in range(8):
        for f in range(8):
            piece = board.board[r, f]
            if piece != 0:
                piece_type = abs(piece)
                if piece_type in piece_planes:
                    plane_idx = piece_planes[piece_type]
                    if piece > 0:  # White piece
                        planes[plane_idx][r, f] = 1.0
                    else:  # Black piece
                        planes[plane_idx + 6][r, f] = 1.0

    # Plane 12: side to move
    stm = 1.0 if board.white_to_move else 0.0
    planes.append(np.full((8, 8), stm, dtype=np.float32))

    # Planes 13-16: castling rights (WK, WQ, BK, BQ)
    for right in board.castling_rights:
        val = 1.0 if right else 0.0
        planes.append(np.full((8, 8), val, dtype=np.float32))

    # Plane 17: en passant target square
    ep_plane = np.zeros((8, 8), dtype=np.float32)
    if board.en_passant_square is not None:
        r, f = board.en_passant_square
        ep_plane[r, f] = 1.0
    planes.append(ep_plane)

    x = np.stack(planes, axis=0)  # (18, 8, 8)
    return torch.from_numpy(x).unsqueeze(0)  # (1, 18, 8, 8)
