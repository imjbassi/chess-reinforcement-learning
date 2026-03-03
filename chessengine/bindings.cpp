```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "board.h"

namespace py = pybind11;

PYBIND11_MODULE(chessengine, m) {
    m.doc() = "Chess engine module for reinforcement learning";

    py::class_<Board>(m, "Board")
        .def(py::init<>(),
             "Construct a new Board with the standard starting position")
        .def("load_fen", &Board::load_fen,
             py::arg("fen"),
             "Load a board position from FEN notation")
        .def("generate_moves", &Board::generate_moves,
             "Generate all legal moves for the current position")
        .def("make_move", &Board::make_move,
             py::arg("uci"),
             "Make a move specified in UCI notation (e.g., 'e2e4')")
        .def("is_game_over", [](Board &b) {
            auto result = b.is_game_over();
            return py::make_tuple(result.first, result.second);
        },
             "Check if the game is over and return (is_over, outcome)")
        .def("pieces", [](const Board &b) {
            // Return bitboard representation of all pieces
            // (6 piece types × 2 colors = 12 bitboards)
            const auto piece_ptr = b.pieces();
            return std::vector<uint64_t>(piece_ptr, piece_ptr + PIECE_NB);
        },
             "Get bitboard representation of all pieces")
        .def("white_to_move", &Board::white_to_move,
             "Return true if it is White's turn to move")
        .def("castling_rights", &Board::castling_rights,
             "Get the current castling rights as a bitmask")
        .def("ep_square", &Board::ep_square,
             "Get the en passant target square, if any")
        .def("export_fen", &Board::export_fen,
             "Export the current position as FEN notation");
}
```