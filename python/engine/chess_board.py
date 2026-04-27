```python
import sys
import os

# Add directories to path for imports
# Project root: needed so Python can find the chessengine .pyd extension
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# python/ directory: needed so `model.model` resolves to python/model/model.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chessengine import Board
from model.model import encode_board


class ChessBoard:
    """
    Wrapper class for the chess engine Board, providing a simplified interface
    for chess game management and state encoding.
    
    This class encapsulates the underlying chess engine and provides methods for:
    - Board initialization and reset
    - Move generation and application
    - Game state queries (game over, legal moves)
    - Board state encoding for machine learning
    """

    def __init__(self, white_to_move: bool = True):
        """
        Initialize a new chess board.

        Args:
            white_to_move: If True, white moves first; otherwise black moves first.
                          Defaults to True (standard chess starting position).
        """
        self._board = Board()
        self._initial_side = "w" if white_to_move else "b"
        self.reset()

    def reset(self):
        """
        Reset the board to the starting position with the configured side to move.
        
        The board is reset to the standard chess starting position with all pieces
        in their initial squares. The side to move is determined by the value set
        during initialization.
        """
        fen = (
            f"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR "
            f"{self._initial_side} KQkq - 0 1"
        )
        self._board.load_fen(fen)

    def get_legal_moves(self):
        """
        Get all legal moves in the current position.

        Returns:
            list: List of legal moves in the current position. Each move is
                  represented in a format determined by the underlying engine.
        """
        return self._board.generate_moves()

    def apply_move(self, move: str):
        """
        Apply a move to the board.

        Args:
            move: The move to apply in string format (e.g., "e2e4" for pawn to e4,
                  "e7e8q" for pawn promotion to queen).
        
        Note:
            This method does not validate whether the move is legal. Ensure the
            move is legal before calling this method to avoid undefined behavior.
        """
        self._board.make_move(move)

    def is_game_over(self):
        """
        Check if the game is over.

        Returns:
            tuple: A tuple of (done, result) where:
                   - done (bool): True if the game is over, False otherwise.
                   - result: The game outcome (e.g., checkmate, stalemate, draw).
                            The exact format depends on the underlying engine.
        """
        return self._board.is_game_over()

    def encode(self):
        """
        Encode the current board state for use in machine learning models.

        Returns:
            Encoded representation of the board state, typically as a numpy array
            or tensor suitable for neural network input. The exact format is
            determined by the encode_board function from the model module.
        """
        return encode_board(self._board)

    def get_fen(self):
        """
        Get the FEN (Forsyth-Edwards Notation) string of the current position.

        Returns:
            str: FEN string representing the current board state, including piece
                 positions, side to move, castling rights, en passant target,
                 halfmove clock, and fullmove number.
        """
        return self._board.export_fen()

    def load_fen(self, fen: str):
        """
        Load a board position from a FEN string.

        Args:
            fen: FEN string representing the desired board state.
        """
        self._board.load_fen(fen)

    def __repr__(self):
        """
        Return a string representation of the ChessBoard instance.

        Returns:
            str: A string showing the current FEN position.
        """
        return f"ChessBoard(fen='{self.get_fen()}')"

    def __str__(self):
        """
        Return a human-readable string representation of the board.

        Returns:
            str: A string showing the current FEN position.
        """
        return self.get_fen()
```