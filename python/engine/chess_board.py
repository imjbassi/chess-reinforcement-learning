import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from chessengine import Board

from model.model import encode_board


class ChessBoard:
    def __init__(self, white_to_move: bool = True):
        # Always use the zero‑arg constructor
        self._b = Board()
        # Store which side to move for resets
        self._side = "w" if white_to_move else "b"

    def reset(self):
        # Include the side in the FEN string
        fen = (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR "
            f"{self._side} KQkq - 0 1"
        )
        self._b.load_fen(fen)

    def get_legal_moves(self):
        return self._b.generate_moves()

    def apply_move(self, move: str):
        self._b.make_move(move)

    def is_game_over(self):
        done, result = self._b.is_game_over()
        return done, result

    def encode(self):
        return encode_board(self._b)

    def get_fen(self):
        return self._b.export_fen()
