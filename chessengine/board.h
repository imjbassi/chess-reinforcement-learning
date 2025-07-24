#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <sstream>
#include <cctype>

using U64 = uint64_t;
enum Piece { WP=0, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK, PIECE_NB };

class Board {
public:
    Board();
    bool load_fen(const std::string &fen);
    std::vector<std::string> generate_moves() const;
    void make_move(const std::string &uci);
    std::pair<bool,int> is_game_over() const;
    std::string export_fen() const;

    // exposed for movegen & checking
    const U64* pieces() const { return pieces_; }
    bool white_to_move() const { return white_to_move_; }
    int  castling_rights() const { return castling_rights_; }
    int  ep_square() const { return ep_square_; }
    void set_white_to_move(bool w); // Add this line
    U64 occupied() const; // Declare occupied() method
    bool in_check(bool white) const;

private:
    U64 pieces_[PIECE_NB];
    bool white_to_move_;
    int  castling_rights_;
    int  ep_square_;
    int  halfmove_clock_, fullmove_number_;

    void clear();
    void parse_placement(const std::string &s);
    void set_piece(Piece p, int sq);

    // Helper function to apply a move without legality validation
    void apply_move_without_validation(const std::string &uci);
};
