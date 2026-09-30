"""Simple Python chess implementation for fallback when C++ engine fails"""
import numpy as np

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
        self.halfmove_clock = 0
        self.reset()

    def reset(self):
        """Reset the board to the initial position."""
        self.board = INITIAL_BOARD.copy()
        self.white_to_move = True
        self.moves_played = 0
        self.castling_rights = [True, True, True, True]  # WK, WQ, BK, BQ
        self.en_passant_square = None
        self.halfmove_clock = 0

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
                self._generate_knight_moves(sq, rank, file, is_white, pseudo_moves)
            elif piece_type == KING:
                self._generate_king_moves(sq, rank, file, piece, is_white, pseudo_moves)
            else:
                self._generate_sliding_moves(sq, rank, file, piece_type, is_white, pseudo_moves)

        # Filter out moves that leave the moving side's king in check
        legal_moves = []
        for uci in pseudo_moves:
            test_board = self._board_after_move(self.board, uci)
            king_val = KING if self.white_to_move else -KING
            king_squares = np.argwhere(test_board == king_val)
            if len(king_squares) == 0:
                continue
            king_sq = coords_to_sq(*king_squares[0])
            if not self._is_attacked_on_board(test_board, king_sq, not self.white_to_move):
                legal_moves.append(uci)

        return legal_moves

    def _board_after_move(self, board, uci):
        """Return a copy of the given board with the UCI move applied (no side/state updates)."""
        new_board = board.copy()
        from_file, from_rank = ord(uci[0]) - ord('a'), int(uci[1]) - 1
        to_file, to_rank = ord(uci[2]) - ord('a'), int(uci[3]) - 1
        piece = new_board[from_rank, from_file]

        # En passant: pawn moves diagonally onto an empty square
        if abs(piece) == PAWN and from_file != to_file and new_board[to_rank, to_file] == 0:
            new_board[from_rank, to_file] = 0

        new_board[from_rank, from_file] = 0

        # Promotion
        if len(uci) == 5:
            promo_type = {'q': QUEEN, 'r': ROOK, 'b': BISHOP, 'n': KNIGHT}[uci[4]]
            piece = promo_type if piece > 0 else -promo_type

        new_board[to_rank, to_file] = piece

        # Castling: move the rook alongside the king
        if abs(piece) == KING and abs(to_file - from_file) == 2:
            rook_from_file = 7 if to_file > from_file else 0
            rook_to_file = 5 if to_file > from_file else 3
            new_board[to_rank, rook_to_file] = new_board[to_rank, rook_from_file]
            new_board[to_rank, rook_from_file] = 0

        return new_board

    def apply_move(self, uci):
        """Apply a UCI move to the board, updating all game state."""
        from_file, from_rank = ord(uci[0]) - ord('a'), int(uci[1]) - 1
        to_file, to_rank = ord(uci[2]) - ord('a'), int(uci[3]) - 1
        piece = self.board[from_rank, from_file]
        is_pawn = abs(piece) == PAWN
        is_capture = self.board[to_rank, to_file] != 0 or \
            (is_pawn and from_file != to_file and self.board[to_rank, to_file] == 0)

        self.board = self._board_after_move(self.board, uci)

        # Update castling rights when king or rooks move (or rooks are captured)
        for idx, corner in enumerate([(0, 7), (0, 0), (7, 7), (7, 0)]):
            if (from_rank, from_file) == corner or (to_rank, to_file) == corner:
                self.castling_rights[idx] = False
        if piece == KING:
            self.castling_rights[0] = self.castling_rights[1] = False
        elif piece == -KING:
            self.castling_rights[2] = self.castling_rights[3] = False

        # Update en passant target square after a double pawn push
        if is_pawn and abs(to_rank - from_rank) == 2:
            self.en_passant_square = ((from_rank + to_rank) // 2, from_file)
        else:
            self.en_passant_square = None

        # Update halfmove clock for the fifty-move rule
        if is_pawn or is_capture:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock = self.halfmove_clock + 1

        self.white_to_move = not self.white_to_move
        self.moves_played += 1

    def is_game_over(self):
        """
        Check whether the game has ended.

        Returns:
            Tuple of (done, result) where result is "1-0", "0-1" or "1/2-1/2".
        """
        # Fifty-move rule (100 plies without a pawn move or capture)
        if self.halfmove_clock >= 100:
            return True, "1/2-1/2"

        # Insufficient material: bare kings
        if np.all((self.board == 0) | (np.abs(self.board) == KING)):
            return True, "1/2-1/2"

        if self.get_legal_moves():
            return False, None

        # No legal moves: checkmate or stalemate
        king_val = KING if self.white_to_move else -KING
        king_squares = np.argwhere(self.board == king_val)
        if len(king_squares) > 0:
            king_sq = coords_to_sq(*king_squares[0])
            if self._is_attacked_on_board(self.board, king_sq, not self.white_to_move):
                return True, "0-1" if self.white_to_move else "1-0"
        return True, "1/2-1/2"

    def get_fen(self):
        """Export the current position as a FEN string."""
        piece_chars = {PAWN: 'P', KNIGHT: 'N', BISHOP: 'B', ROOK: 'R', QUEEN: 'Q', KING: 'K'}
        rows = []
        for rank in range(7, -1, -1):
            row = ""
            empty = 0
            for file in range(8):
                piece = self.board[rank, file]
                if piece == 0:
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                c = piece_chars[abs(piece)]
                row += c if piece > 0 else c.lower()
            if empty:
                row += str(empty)
            rows.append(row)

        stm = 'w' if self.white_to_move else 'b'
        cast = "".join(c for c, right in zip("KQkq", self.castling_rights) if right) or "-"
        if self.en_passant_square is not None:
            ep_rank, ep_file = self.en_passant_square
            ep = f"{chr(ep_file + ord('a'))}{ep_rank + 1}"
        else:
            ep = "-"
        halfmove = self.halfmove_clock
        fullmove = self.moves_played // 2 + 1
        return f"{'/'.join(rows)} {stm} {cast} {ep} {halfmove} {fullmove}"


# Plane index for each piece type, matching the C++ engine's bitboard order:
# WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK
_PLANE_OF_TYPE = {PAWN: 0, KNIGHT: 1, BISHOP: 2, ROOK: 3, QUEEN: 4, KING: 5}


def encode_simple_board(board):
    """
    Encode a SimpleChessBoard into the 18-plane tensor the network expects.

    Plane layout matches model.encode_board:
    - Planes 0-11: piece positions (WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK)
    - Plane 12: side to move (all ones when white to move)
    - Planes 13-16: castling rights (WK, WQ, BK, BQ)
    - Plane 17: en passant target square

    Returns:
        torch.Tensor of shape (1, 18, 8, 8)
    """
    import torch

    planes = np.zeros((18, 8, 8), dtype=np.float32)

    for rank in range(8):
        for file in range(8):
            piece = board.board[rank, file]
            if piece == 0:
                continue
            plane = _PLANE_OF_TYPE[abs(piece)] + (0 if piece > 0 else 6)
            planes[plane, rank, file] = 1.0

    planes[12, :, :] = 1.0 if board.white_to_move else 0.0

    for i, right in enumerate(board.castling_rights):
        if right:
            planes[13 + i, :, :] = 1.0

    if board.en_passant_square is not None:
        ep_rank, ep_file = board.en_passant_square
        planes[17, ep_rank, ep_file] = 1.0

    return torch.from_numpy(planes).unsqueeze(0)
