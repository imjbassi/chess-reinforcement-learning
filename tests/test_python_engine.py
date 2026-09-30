"""Correctness tests for the pure-Python fallback engine."""
import copy

import numpy as np
import pytest

from python_chess import SimpleChessBoard, encode_simple_board

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def perft(board, depth):
    if depth == 0:
        return 1
    total = 0
    for move in board.get_legal_moves():
        child = copy.deepcopy(board)
        child.apply_move(move)
        total += perft(child, depth - 1)
    return total


def test_starting_fen():
    assert SimpleChessBoard().get_fen() == STARTING_FEN


@pytest.mark.parametrize("depth,expected", [(1, 20), (2, 400), (3, 8902)])
def test_perft_startpos(depth, expected):
    assert perft(SimpleChessBoard(), depth) == expected


def test_scholars_mate():
    board = SimpleChessBoard()
    for move in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]:
        board.apply_move(move)
    done, result = board.is_game_over()
    assert done
    assert result == "1-0"


def test_stalemate():
    board = SimpleChessBoard()
    # Fastest known stalemate (Sam Loyd), 10 moves
    for move in ["e2e3", "a7a5", "d1h5", "a8a6", "h5a5", "h7h5", "h2h4", "a6h6",
                 "a5c7", "f7f6", "c7d7", "e8f7", "d7b7", "d8d3", "b7b8", "d3h7",
                 "b8c8", "f7g6", "c8e6"]:
        board.apply_move(move)
    done, result = board.is_game_over()
    assert done
    assert result == "1/2-1/2"


def test_en_passant_capture_removes_pawn():
    board = SimpleChessBoard()
    for move in ["e2e4", "a7a6", "e4e5", "d7d5"]:
        board.apply_move(move)
    assert "e5d6" in board.get_legal_moves()
    board.apply_move("e5d6")
    # The captured d5 pawn must be gone
    assert board.board[4, 3] == 0


def test_castling_moves_rook():
    board = SimpleChessBoard()
    for move in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]:
        board.apply_move(move)
    assert "e1g1" in board.get_legal_moves()
    board.apply_move("e1g1")
    fen = board.get_fen().split()[0]
    assert fen.endswith("RNBQ1RK1")


def test_promotion():
    board = SimpleChessBoard()
    board.board[:] = 0
    board.board[0, 4] = 6   # white king e1
    board.board[7, 0] = -6  # black king a8
    board.board[6, 7] = 1   # white pawn h7
    board.castling_rights = [False] * 4
    moves = board.get_legal_moves()
    assert "h7h8q" in moves and "h7h8n" in moves
    board.apply_move("h7h8q")
    assert board.board[7, 7] == 5  # queen


def test_fifty_move_rule():
    board = SimpleChessBoard()
    board.halfmove_clock = 100
    done, result = board.is_game_over()
    assert done and result == "1/2-1/2"


def test_encode_simple_board_shape_and_planes():
    torch = pytest.importorskip("torch")
    board = SimpleChessBoard()
    t = encode_simple_board(board)
    assert t.shape == (1, 18, 8, 8)
    x = t.numpy()[0]
    assert x[0].sum() == 8   # white pawns
    assert x[6].sum() == 8   # black pawns
    assert x[5, 0, 4] == 1.0  # white king on e1
    assert np.all(x[12] == 1.0)  # white to move
    assert np.all(x[13:17] == 1.0)  # full castling rights
    assert x[17].sum() == 0  # no en passant


def test_random_playthrough_against_python_chess():
    chess = pytest.importorskip("chess")
    import random

    rng = random.Random(7)
    for _ in range(3):
        ref = chess.Board()
        mine = SimpleChessBoard()
        for _ in range(80):
            ref_moves = sorted(m.uci() for m in ref.legal_moves)
            my_moves = sorted(mine.get_legal_moves())
            assert ref_moves == my_moves, f"mismatch at {ref.fen()}"
            if not ref_moves:
                break
            move = rng.choice(ref_moves)
            ref.push_uci(move)
            mine.apply_move(move)
