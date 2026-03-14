// chessengine/movegen.cpp
#include "movegen.h"
#include "board.h"
#include <intrin.h>
#include <vector>
#include <array>
#include <cstdint>
#include <string>

// Pop the least-significant 1-bit from bb and return its index 0..63
static inline int pop_lsb(uint64_t &bb) {
    unsigned long idx;
    _BitScanForward64(&idx, bb);
    bb &= bb - 1;
    return static_cast<int>(idx);
}

// Convert square index (0..63) to UCI string like "e2"
static inline std::string sq_to_uci(int sq) {
    char file = 'a' + (sq & 7);
    char rank = '1' + (sq >> 3);
    return std::string{file, rank};
}

// Append move "e2e4" or promotion "e7e8q"
static inline void push_move(std::vector<std::string> &out, int from, int to, char promo = '\0') {
    auto uci = sq_to_uci(from) + sq_to_uci(to);
    if (promo) uci.push_back(promo);
    out.push_back(uci);
}

// King one-step directions
constexpr std::array<int, 8> KING_DIRS = {+1, -1, +8, -8, +9, +7, -7, -9};
constexpr std::array<int, 4> ROOK_DIRS = {+8, -8, +1, -1};
constexpr std::array<int, 4> BISHOP_DIRS = {+9, +7, -9, -7};

// Compute knight attack bitboard from a given square
static inline uint64_t knight_attacks(int sq) {
    uint64_t b = 1ULL << sq;
    uint64_t l1 = (b >> 1) & 0x7f7f7f7f7f7f7f7fULL;
    uint64_t l2 = (b >> 2) & 0x3f3f3f3f3f3f3f3fULL;
    uint64_t r1 = (b << 1) & 0xfefefefefefefefeULL;
    uint64_t r2 = (b << 2) & 0xfcfcfcfcfcfcfcfcULL;
    uint64_t h1 = l1 | r1;
    uint64_t h2 = l2 | r2;
    return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
}

// Check if a square is attacked by the opponent
static bool is_square_attacked(int sq, bool by_white, const uint64_t *pieces, uint64_t occ) {
    // Pawn attacks
    int pawn_offset1 = by_white ? 7 : -7;
    int pawn_offset2 = by_white ? 9 : -9;
    int pawn_sq1 = sq - pawn_offset1;
    int pawn_sq2 = sq - pawn_offset2;
    
    if (pawn_sq1 >= 0 && pawn_sq1 < 64 && abs((sq & 7) - (pawn_sq1 & 7)) == 1) {
        if ((1ULL << pawn_sq1) & pieces[by_white ? WP : BP]) return true;
    }
    if (pawn_sq2 >= 0 && pawn_sq2 < 64 && abs((sq & 7) - (pawn_sq2 & 7)) == 1) {
        if ((1ULL << pawn_sq2) & pieces[by_white ? WP : BP]) return true;
    }
    
    // Knight attacks
    uint64_t knights = pieces[by_white ? WN : BN];
    if (knight_attacks(sq) & knights) return true;
    
    // Sliding pieces (rooks, bishops, queens)
    int sq_file = sq & 7;
    int sq_rank = sq >> 3;
    
    // Rook-like moves (horizontal/vertical)
    for (int d : ROOK_DIRS) {
        int t = sq + d;
        int t_file = t & 7;
        int t_rank = t >> 3;
        
        while (t >= 0 && t < 64) {
            // Check for board wrap
            if (d == +1 || d == -1) {
                if (t_rank != sq_rank) break;
            }
            
            uint64_t bit = 1ULL << t;
            if (bit & occ) {
                if (bit & (pieces[by_white ? WR : BR] | pieces[by_white ? WQ : BQ])) return true;
                break;
            }
            t += d;
            t_file = t & 7;
            t_rank = t >> 3;
        }
    }
    
    // Bishop-like moves (diagonal)
    for (int d : BISHOP_DIRS) {
        int t = sq + d;
        int t_file = t & 7;
        int t_rank = t >> 3;
        
        while (t >= 0 && t < 64) {
            // Check for board wrap
            if (abs(t_file - sq_file) != abs(t_rank - sq_rank)) break;
            
            uint64_t bit = 1ULL << t;
            if (bit & occ) {
                if (bit & (pieces[by_white ? WB : BB] | pieces[by_white ? WQ : BQ])) return true;
                break;
            }
            t += d;
            t_file = t & 7;
            t_rank = t >> 3;
        }
    }
    
    // King proximity
    uint64_t opp_king = pieces[by_white ? WK : BK];
    if (opp_king) {
        unsigned long idx;
        _BitScanForward64(&idx, opp_king);
        int king_sq = static_cast<int>(idx);
        int dx = abs((king_sq & 7) - sq_file);
        int dy = abs((king_sq >> 3) - sq_rank);
        if (dx <= 1 && dy <= 1) return true;
    }
    
    return false;
}

std::vector<std::string> generate_pseudo_legal_moves(const Board &b) {
    std::vector<std::string> moves;
    moves.reserve(128); // Reserve space for typical move count
    
    const uint64_t *P = b.pieces();
    bool W = b.white_to_move();
    
    // Compute occupancy bitboards
    uint64_t occ = 0, own = 0, opp = 0;
    for (int i = 0; i < PIECE_NB; ++i) occ |= P[i];
    if (W) {
        for (int i = WP; i <= WK; ++i) own |= P[i];
        for (int i = BP; i <= BK; ++i) opp |= P[i];
    } else {
        for (int i = BP; i <= BK; ++i) own |= P[i];
        for (int i = WP; i <= WK; ++i) opp |= P[i];
    }
    uint64_t empty = ~occ;

    // 1) Pawn pushes (single, double, promotions)
    {
        uint64_t pawns = P[W ? WP : BP];
        
        // Single-push
        uint64_t one = W ? ((pawns << 8) & empty) : ((pawns >> 8) & empty);
        
        // Double-push from starting rank
        uint64_t two = 0ULL;
        if (W) {
            uint64_t rank2 = pawns & 0x000000000000FF00ULL;
            two = ((rank2 << 16) & empty & (empty << 8));
        } else {
            uint64_t rank7 = pawns & 0x00FF000000000000ULL;
            two = ((rank7 >> 16) & empty & (empty >> 8));
        }
        
        // Process single pushes
        uint64_t tmp = one;
        while (tmp) {
            int to = pop_lsb(tmp);
            int from = W ? to - 8 : to + 8;
            bool promo_rank = (W && to >= 56) || (!W && to <= 7);
            if (promo_rank) {
                for (char pr : {'q', 'r', 'b', 'n'})
                    push_move(moves, from, to, pr);
            } else {
                push_move(moves, from, to);
            }
        }
        
        // Process double pushes
        tmp = two;
        while (tmp) {
            int to = pop_lsb(tmp);
            int from = W ? to - 16 : to + 16;
            push_move(moves, from, to);
        }
    }

    // 2) Pawn captures (with promotions and en passant)
    {
        uint64_t pawns = P[W ? WP : BP];
        
        // Compute capture bitboards
        uint64_t cap_left = W
            ? ((pawns << 7) & ~0x0101010101010101ULL)
            : ((pawns >> 9) & ~0x0101010101010101ULL);
        uint64_t cap_right = W
            ? ((pawns << 9) & ~0x8080808080808080ULL)
            : ((pawns >> 7) & ~0x8080808080808080ULL);
        uint64_t caps_left = cap_left & opp;
        uint64_t caps_right = cap_right & opp;

        auto process_captures = [&](uint64_t caps, int delta_white, int delta_black) {
            uint64_t tmp = caps;
            while (tmp) {
                int to = pop_lsb(tmp);
                int from = W ? to - delta_white : to + delta_black;
                bool promo_rank = (W && to >= 56) || (!W && to <= 7);
                if (promo_rank) {
                    for (char pr : {'q', 'r', 'b', 'n'})
                        push_move(moves, from, to, pr);
                } else {
                    push_move(moves, from, to);
                }
            }
        };
        
        process_captures(caps_left, 7, 9);
        process_captures(caps_right, 9, 7);

        // En passant
        int ep = b.ep_square();
        if (ep >= 0) {
            int ep_rank = ep >> 3;
            int ep_file = ep & 7;
            if (W && ep_rank == 5) {
                if (ep_file > 0) {
                    int from1 = ep - 9;
                    if ((pawns >> from1) & 1) push_move(moves, from1, ep);
                }
                if (ep_file < 7) {
                    int from2 = ep - 7;
                    if ((pawns >> from2) & 1) push_move(moves, from2, ep);
                }
            } else if (!W && ep_rank == 2) {
                if (ep_file < 7) {
                    int from1 = ep + 9;
                    if ((pawns >> from1) & 1) push_move(moves, from1, ep);
                }
                if (ep_file > 0) {
                    int from2 = ep + 7;
                    if ((pawns >> from2) & 1) push_move(moves, from2, ep);
                }
            }
        }
    }

    // 3) Knight moves
    {
        uint64_t knights = P[W ? WN : BN];
        uint64_t tmp = knights;
        while (tmp) {
            int sq = pop_lsb(tmp);
            uint64_t attacks = knight_attacks(sq) & ~own;
            while (attacks) {
                int to = pop_lsb(attacks);
                push_move(moves, sq, to);
            }
        }
    }

    // 4) Sliding pieces (Rooks, Bishops, Queens)
    auto generate_sliding_moves = [&](uint64_t pieces_bb, const std::array<int, 4> &dirs) {
        while (pieces_bb) {
            int sq = pop_lsb(pieces_bb);
            int sq_rank = sq >> 3;
            int sq_file = sq & 7;
            
            for (int d : dirs) {
                int t = sq + d;
                int t_rank = t >> 3;
                int t_file = t & 7;
                
                while (t >= 0 && t < 64) {
                    // Check for board wrap
                    if (d == +1 || d == -1) {
                        if (t_rank != sq_rank) break;
                    } else if (d == +9 || d == -7) {
                        if (t_file - sq_file != t_rank - sq_rank) break;
                    } else if (d == +7 || d == -9) {
                        if (t_file - sq_file != sq_rank - t_rank) break;
                    }
                    
                    if ((own >> t) & 1) break;
                    push_move(moves, sq, t);
                    if ((occ >> t) & 1) break;
                    
                    t += d;
                    t_rank = t >> 3;
                    t_file = t & 7;
                }
            }
        }
    };

    // Rooks
    generate_sliding_moves(P[W ? WR : BR], ROOK_DIRS);
    
    // Bishops
    generate_sliding_moves(P[W ? WB : BB], BISHOP_DIRS);
    
    // Queens (combine rook and bishop directions)
    {
        uint64_t queens = P[W ? WQ : BQ];
        generate_sliding_moves(queens, ROOK_DIRS);
        queens = P[W ? WQ : BQ]; // Reset bitboard
        generate_sliding_moves(queens, BISHOP_DIRS);
    }

    // 5) King moves + castling
    {
        uint64_t king_bb = P[W ? WK : BK];
        if (!king_bb) return moves; // No king, return early
        
        int king_sq = pop_lsb(king_bb);
        int king_file = king_sq & 7;
        
        // One-square moves (with attack validation)
        for (int d : KING_DIRS) {
            int to = king_sq + d;
            if (to < 0 || to >= 64) continue;
            if (abs((to & 7) - king_file) > 1) continue;
            if ((own >> to) & 1) continue;
            if (!is_square_attacked(to, !W, P, occ)) {
                push_move(moves, king_sq, to);
            }
        }
        
        // Castling
        int rights = b.castling_rights();
        if (W) {
            // White kingside (e1-g1)