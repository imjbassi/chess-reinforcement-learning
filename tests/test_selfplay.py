"""Tests for self-play move selection and encoding plumbing."""
import pytest

torch = pytest.importorskip("torch")

from python_chess import SimpleChessBoard, encode_simple_board
from selfplay import _uci_to_index, _index_to_uci, _select_move_from_policy


def test_uci_index_roundtrip():
    for uci in ["e2e4", "a1h8", "h7h8", "g8f6"]:
        assert _index_to_uci(_uci_to_index(uci)) == uci


def test_select_move_returns_legal_move():
    board = SimpleChessBoard()
    legal = board.get_legal_moves()
    logits = torch.zeros(1, 4096)
    for _ in range(20):
        uci, probs = _select_move_from_policy(logits, legal, temperature=1.0)
        assert uci in legal
        assert probs.shape == (1, 4096)


def test_select_move_handles_promotions():
    board = SimpleChessBoard()
    board.board[:] = 0
    board.board[0, 4] = 6   # white king e1
    board.board[7, 0] = -6  # black king a8
    board.board[6, 7] = 1   # white pawn h7
    board.castling_rights = [False] * 4
    legal = board.get_legal_moves()
    # Bias the policy entirely toward the promotion square
    logits = torch.full((1, 4096), -1e9)
    logits[0, _uci_to_index("h7h8")] = 1e9
    uci, _ = _select_move_from_policy(logits, legal, temperature=1.0)
    assert uci in legal
    assert uci.startswith("h7h8")


def test_model_forward_shapes():
    from model.model import ChessNet

    net = ChessNet()
    net.eval()
    state = encode_simple_board(SimpleChessBoard())
    with torch.no_grad():
        policy, value = net(state)
    assert policy.shape == (1, 4096)
    assert value.shape == (1, 1)
    assert -1.0 <= value.item() <= 1.0
