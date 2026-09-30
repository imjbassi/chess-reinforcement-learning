"""Correctness tests for the C++ chess engine (skipped when it isn't built)."""
import pytest

chessengine = pytest.importorskip("chessengine")

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
POSITION5_FEN = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"


def make_board(fen):
    board = chessengine.Board()
    board.load_fen(fen)
    return board


def perft(fen, depth):
    board = make_board(fen)
    if depth == 0:
        return 1
    total = 0
    for move in board.generate_moves():
        child = make_board(fen)
        child.make_move(move)
        total += perft(child.export_fen(), depth - 1)
    return total


def test_fen_roundtrip():
    for fen in (STARTING_FEN, KIWIPETE_FEN, POSITION5_FEN):
        assert make_board(fen).export_fen() == fen


@pytest.mark.parametrize("depth,expected", [(1, 20), (2, 400), (3, 8902), (4, 197281)])
def test_perft_startpos(depth, expected):
    assert perft(STARTING_FEN, depth) == expected


@pytest.mark.parametrize("depth,expected", [(1, 48), (2, 2039), (3, 97862)])
def test_perft_kiwipete(depth, expected):
    assert perft(KIWIPETE_FEN, depth) == expected


@pytest.mark.parametrize("depth,expected", [(1, 44), (2, 1486), (3, 62379)])
def test_perft_position5(depth, expected):
    assert perft(POSITION5_FEN, depth) == expected


def test_illegal_move_raises():
    board = chessengine.Board()
    with pytest.raises(RuntimeError):
        board.make_move("e2e5")


def test_scholars_mate():
    board = chessengine.Board()
    for move in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]:
        board.make_move(move)
    done, result = board.is_game_over()
    assert done
    assert result == 1  # white wins


def test_pieces_and_state_accessors():
    board = chessengine.Board()
    bitboards = board.pieces()
    assert len(bitboards) == 12
    assert bin(bitboards[0]).count("1") == 8  # eight white pawns
    assert board.white_to_move()
    assert board.castling_rights() == 0b1111
    assert board.ep_square() == -1
    board.make_move("e2e4")
    assert board.ep_square() == 20  # e3
