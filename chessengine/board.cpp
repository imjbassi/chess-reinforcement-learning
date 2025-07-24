#include "board.h"
#include "movegen.h"
#include "attack_tables.h"
#include <algorithm>
#include <intrin.h>
#include <cassert>
#include <sstream>
#include <cctype>
#include <functional>

static inline int pop_lsb(U64 &bb) {
    unsigned long idx;
    _BitScanForward64(&idx, bb);
    bb &= bb - 1;
    return int(idx);
}

static inline int uci_sq(const std::string &u, int i) {
    return (u[i+1] - '1')*8 + (u[i] - 'a');
}

Board::Board() {
    load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Board::clear() {
    std::fill(std::begin(pieces_), std::end(pieces_), 0ULL);
    white_to_move_   = true;
    castling_rights_ = 1|2|4|8;
    ep_square_       = -1;
    halfmove_clock_  = 0;
    fullmove_number_ = 1;
}

bool Board::load_fen(const std::string &fen) {
    clear();
    std::istringstream ss(fen);
    std::string pl, stm, cast, ep;
    if (!(ss >> pl >> stm >> cast >> ep >> halfmove_clock_ >> fullmove_number_))
        return false;
    parse_placement(pl);
    white_to_move_   = (stm == "w");
    castling_rights_ = 0;
    if (cast.find('K')!=std::string::npos) castling_rights_|=1;
    if (cast.find('Q')!=std::string::npos) castling_rights_|=2;
    if (cast.find('k')!=std::string::npos) castling_rights_|=4;
    if (cast.find('q')!=std::string::npos) castling_rights_|=8;
    if (ep != "-") {
        int f = ep[0]-'a', r = ep[1]-'1';
        ep_square_ = r*8 + f;
    }
    return true;
}

void Board::parse_placement(const std::string &s) {
    int rank = 7, file = 0;
    for (char c : s) {
        if (c == '/') {
            --rank;
            file = 0;
        } else if (std::isdigit(c)) {
            file += c - '0';
        } else {
            int sq = rank*8 + file++;
            switch (c) {
                case 'P': set_piece(WP, sq); break;
                case 'N': set_piece(WN, sq); break;
                case 'B': set_piece(WB, sq); break;
                case 'R': set_piece(WR, sq); break;
                case 'Q': set_piece(WQ, sq); break;
                case 'K': set_piece(WK, sq); break;
                case 'p': set_piece(BP, sq); break;
                case 'n': set_piece(BN, sq); break;
                case 'b': set_piece(BB, sq); break;
                case 'r': set_piece(BR, sq); break;
                case 'q': set_piece(BQ, sq); break;
                case 'k': set_piece(BK, sq); break;
            }
        }
    }
}

void Board::set_piece(Piece p, int sq) {
    pieces_[p] |= (1ULL << sq);
}

std::vector<std::string> Board::generate_moves() const {
    try {
        auto pseudo = generate_pseudo_legal_moves(*this);
        std::vector<std::string> legal;
        
        for (const auto &m : pseudo) {
            // Make a copy of the board and try the move
            Board tmp = *this;
            try {
                // Apply move without legality check to prevent recursion
                tmp.apply_move_without_validation(m);
                // Now check if the resulting position leaves king in check
                if (!tmp.in_check(!tmp.white_to_move_)) {
                    legal.push_back(m);
                }
            } catch (...) {
                // Invalid move, skip it
                continue;
            }
        }
        return legal;
    } catch (...) {
        // If anything fails, return empty list instead of hanging
        return {};
    }
}

// Add this new function to avoid recursion
void Board::apply_move_without_validation(const std::string &uci) {
    assert(uci.size() == 4 || uci.size() == 5);
    int from = uci_sq(uci, 0);
    int to   = uci_sq(uci, 2);

    // Rest of the code identical to make_move except without the legal move check
    U64 occ = 0;
    for (int i = 0; i < PIECE_NB; ++i) occ |= pieces_[i];

    Piece promo = static_cast<Piece>(PIECE_NB);
    if (uci.size() == 5) {
        char p = std::tolower(uci[4]);
        if      (p=='q') promo = white_to_move_ ? WQ : BQ;
        else if (p=='r') promo = white_to_move_ ? WR : BR;
        else if (p=='b') promo = white_to_move_ ? WB : BB;
        else if (p=='n') promo = white_to_move_ ? WN : BN;
    }

    U64 fb = 1ULL<<from, tb = 1ULL<<to;
    bool pawn = false, cap = false;

    for (int p = white_to_move_ ? WP : BP; p <= (white_to_move_ ? WK : BK); ++p) {
        if (pieces_[p] & fb) {
            if (p==WP||p==BP) pawn = true;
            pieces_[p] &= ~fb;
            for (int q = 0; q < PIECE_NB; ++q) {
                if (pieces_[q] & tb) {
                    pieces_[q] &= ~tb;
                    cap = true;
                }
            }
            if (pawn && to == ep_square_ && !(occ & tb)) {
                int cs = white_to_move_ ? to-8 : to+8;
                U64 cb = 1ULL<<cs;
                for (int q = BP; q <= BK; ++q) {
                    if (pieces_[q] & cb) {
                        pieces_[q] &= ~cb;
                        cap = true;
                        break;
                    }
                }
            }
            int dst = (promo != PIECE_NB ? promo : p);
            pieces_[dst] |= tb;
            break;
        }
    }

    if (!pawn && ((white_to_move_ && from==4 && (to==6||to==2)) ||
                  (!white_to_move_&& from==60 && (to==62||to==58)))) {
        if (white_to_move_) {
            if (to==6) {
                pieces_[WR] &= ~(1ULL<<7);
                pieces_[WR] |=  (1ULL<<5);
            } else {
                pieces_[WR] &= ~(1ULL<<0);
                pieces_[WR] |=  (1ULL<<3);
            }
        } else {
            if (to==62) {
                pieces_[BR] &= ~(1ULL<<63);
                pieces_[BR] |=  (1ULL<<61);
            } else {
                pieces_[BR] &= ~(1ULL<<56);
                pieces_[BR] |=  (1ULL<<59);
            }
        }
    }

    if (white_to_move_) {
        if (fb & (1ULL<<4)) castling_rights_ &= ~(1|2);
        if (from==0||to==0) castling_rights_ &= ~2;
        if (from==7||to==7) castling_rights_ &= ~1;
    } else {
        if (fb & (1ULL<<60)) castling_rights_ &= ~(4|8);
        if (from==56||to==56) castling_rights_ &= ~8;
        if (from==63||to==63) castling_rights_ &= ~4;
    }

    if (pawn && !cap && std::abs(to-from)==16)
        ep_square_ = white_to_move_ ? from+8 : from-8;
    else
        ep_square_ = -1;

    if (pawn || cap) halfmove_clock_ = 0;
    else             ++halfmove_clock_;

    if (!white_to_move_) ++fullmove_number_;

    white_to_move_ = !white_to_move_;

    assert(pieces_[white_to_move_ ? WK : BK] != 0 && "King was captured!");
}

std::pair<bool,int> Board::is_game_over() const {
    if (white_to_move_) {
        if (pieces_[WK] == 0ULL) return {true, -1};
    } else {
        if (pieces_[BK] == 0ULL) return {true, +1};
    }

    auto m = generate_moves();
    if (!m.empty()) return {false,0};
    bool chk = in_check(white_to_move_);
    return {true, chk ? (white_to_move_ ? -1 : +1) : 0};
}

void Board::set_white_to_move(bool w) {
    white_to_move_ = w;
}

std::string Board::export_fen() const {
    // Minimal stub: return a placeholder or reconstruct FEN if you have the logic
    return "FEN export not implemented";
}

// Add after apply_move_without_validation function:

bool Board::in_check(bool white) const {
    // Find the king
    U64 king_bb = pieces_[white ? WK : BK];
    if (!king_bb) return false;  // No king (shouldn't happen in normal chess)

    // Get king square
    unsigned long idx;
    _BitScanForward64(&idx, king_bb);
    int king_sq = static_cast<int>(idx);

    // Calculate full occupancy for sliding piece checks
    U64 occ = occupied();
    
    // Check for knight attacks
    if (knight_attacks(king_sq) & pieces_[white ? BN : WN])
        return true;
        
    // Check for pawn attacks
    if (pawn_attacks(king_sq, white) & pieces_[white ? BP : WP])
        return true;
        
    // Check for king proximity (adjacent kings)
    if (king_attacks(king_sq) & pieces_[white ? BK : WK])
        return true;
        
    // Check for bishop/queen diagonal attacks
    if (bishop_attacks(king_sq, occ) & (pieces_[white ? BB : WB] | pieces_[white ? BQ : WQ]))
        return true;
        
    // Check for rook/queen orthogonal attacks
    if (rook_attacks(king_sq, occ) & (pieces_[white ? BR : WR] | pieces_[white ? BQ : WQ]))
        return true;
        
    return false;
}

U64 Board::occupied() const {
    U64 occ = 0;
    for (int i = 0; i < PIECE_NB; ++i)
        occ |= pieces_[i];
    return occ;
}

void Board::make_move(const std::string &uci) {
    assert(uci.size() == 4 || uci.size() == 5);
    
    // Check if move is legal by generating all legal moves
    auto legal_moves = generate_moves();
    if (std::find(legal_moves.begin(), legal_moves.end(), uci) == legal_moves.end()) {
        throw std::runtime_error("Illegal move: " + uci);
    }
    
    // Apply the move since it's legal
    apply_move_without_validation(uci);
}
