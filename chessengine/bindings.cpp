#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "board.h"

namespace py = pybind11;

PYBIND11_MODULE(chessengine, m) {
    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def("load_fen", &Board::load_fen, py::arg("fen"))
        .def("generate_moves", &Board::generate_moves)
        .def("make_move", &Board::make_move, py::arg("uci"))
        .def("is_game_over", [](Board &b) {
            auto pr = b.is_game_over();
            return py::make_tuple(pr.first, pr.second);
        })
        // ───────── expose internals ─────────
        .def("pieces", [](Board &b){
            // copy 12 bitboards into a Python list<uint64_t>
            auto p = b.pieces();
            return std::vector<uint64_t>(p, p + PIECE_NB);
        })
        .def("white_to_move", &Board::white_to_move)
        .def("castling_rights", &Board::castling_rights)
        .def("ep_square", &Board::ep_square)
        .def("export_fen", &Board::export_fen)
    ;
}