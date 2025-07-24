// chessengine/movegen.cpp
#include "movegen.h"
#include "board.h"
#include <intrin.h>
#include <vector>
#include <array>
#include <cstdint>
#include <string>

// pop the least‑significant 1‑bit from bb and return its index 0..63
static inline int pop_lsb(uint64_t &bb) {
    unsigned long idx;
    _BitScanForward64(&idx, bb);
    bb &= bb - 1;
    return static_cast<int>(idx);
}

// square index (0..63) → UCI string like "e2"
static inline std::string sq_to_uci(int sq) {
    char file = 'a' + (sq & 7);
    char rank = '1' + (sq >> 3);
    return std::string{file, rank};
}

// append move "e2e4" or promotion "e7e8q"
static inline void push_move(std::vector<std::string> &out, int f, int t, char promo = '\0') {
    auto u = sq_to_uci(f) + sq_to_uci(t);
    if (promo) u.push_back(promo);
    out.push_back(u);
}

// King one‑step directions
constexpr std::array<int,8> KingD = {+1,-1,+8,-8,+9,+7,-7,-9};

std::vector<std::string> generate_pseudo_legal_moves(const Board &b) {
    std::vector<std::string> moves;
    const uint64_t *P = b.pieces();
    bool W = b.white_to_move();
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
        uint64_t paw = P[W ? WP : BP];
        // single‑push
        uint64_t one = W ? ((paw << 8) & empty)
                         : ((paw >> 8) & empty);
        // double‑push from rank 2 (white) or rank 7 (black)
        uint64_t two = 0ULL;
        if (W) {
            uint64_t rank2 = paw & 0x000000000000FF00ULL;
            two = ((rank2 << 16) & empty & (empty << 8));
        } else {
            uint64_t rank7 = paw & 0x00FF000000000000ULL;
            two = ((rank7 >> 16) & empty & (empty >> 8));
        }
        // single pushes
        uint64_t tmp = one;
        while (tmp) {
            int to = pop_lsb(tmp);
            int from = W ? to - 8 : to + 8;
            bool promo_rank = (W && to/8 == 7) || (!W && to/8 == 0);
            if (promo_rank) {
                for (char pr : {'q','r','b','n'})
                    push_move(moves, from, to, pr);
            } else {
                push_move(moves, from, to);
            }
        }
        // double pushes (never promotions)
        tmp = two;
        while (tmp) {
            int to = pop_lsb(tmp);
            int from = W ? to - 16 : to + 16;
            push_move(moves, from, to);
        }
    }

    // 2) Pawn captures (with promotions and en passant)
    {
        uint64_t paw = P[W ? WP : BP];
        uint64_t capL = W
            ? ((paw << 7) & ~0x0101010101010101ULL)
            : ((paw >> 9) & ~0x0101010101010101ULL);
        uint64_t capR = W
            ? ((paw << 9) & ~0x8080808080808080ULL)
            : ((paw >> 7) & ~0x8080808080808080ULL);
        uint64_t capsL = capL & opp;
        uint64_t capsR = capR & opp;

        auto do_caps = [&](uint64_t C, int delta, int delta_alt){
            uint64_t tmp = C;
            while(tmp) {
                int to = pop_lsb(tmp);
                int from = W ? to - delta : to + delta_alt;
                bool promo_rank = (W && to/8 == 7) || (!W && to/8 == 0);
                if (promo_rank) {
                    for(char pr : {'q','r','b','n'})
                        push_move(moves, from, to, pr);
                } else {
                    push_move(moves, from, to);
                }
            }
        };
        do_caps(capsL, 7, 9);
        do_caps(capsR, 9, 7);

        // En passant
        int ep = b.ep_square();
        if (ep >= 0) {
            int ep_rank = ep / 8;
            int ep_file = ep % 8;
            if (W) {
                // White pawn can capture en passant from rank 5
                if (ep_rank == 5) {
                    uint64_t pawns = paw & ((1ULL << (ep - 9)) | (1ULL << (ep - 7)));
                    if (pawns & (1ULL << (ep - 9))) push_move(moves, ep - 9, ep);
                    if (pawns & (1ULL << (ep - 7))) push_move(moves, ep - 7, ep);
                }
            } else {
                // Black pawn can capture en passant from rank 4
                if (ep_rank == 2) {
                    uint64_t pawns = paw & ((1ULL << (ep + 9)) | (1ULL << (ep + 7)));
                    if (pawns & (1ULL << (ep + 9))) push_move(moves, ep + 9, ep);
                    if (pawns & (1ULL << (ep + 7))) push_move(moves, ep + 7, ep);
                }
            }
        }
    }

    // 3) Knight moves
    {
        auto knight_attacks = [&](int sq)->uint64_t {
            uint64_t b = 1ULL << sq;
            uint64_t l1 = (b >> 1) & 0x7f7f7f7f7f7f7f7fULL;
            uint64_t l2 = (b >> 2) & 0x3f3f3f3f3f3f3f3fULL;
            uint64_t r1 = (b << 1) & 0xfefefefefefefefeULL;
            uint64_t r2 = (b << 2) & 0xfcfcfcfcfcfcfcfcULL;
            uint64_t h1 = l1 | r1;
            uint64_t h2 = l2 | r2;
            return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
        };
        uint64_t kn = P[W ? WN : BN];
        uint64_t tmp = kn;
        while (tmp) {
            int sq = pop_lsb(tmp);
            uint64_t att = knight_attacks(sq) & ~own;
            while (att) {
                int to = pop_lsb(att);
                push_move(moves, sq, to);
            }
        }
    }

    // 4) Sliding pieces
    constexpr std::array<int,4> rook_dirs   = { +8, -8, +1, -1 };
    constexpr std::array<int,4> bishop_dirs = { +9, +7, -9, -7 };

    // Rooks - FIXED
    {
        uint64_t bb = P[W ? WR : BR];
        while (bb) {
            int sq = pop_lsb(bb);
            int r0 = sq / 8;  // starting rank
            int f0 = sq % 8;  // starting file
            
            for (int d : rook_dirs) {
                int t = sq + d;
                int r = r0 + (d == 8 ? 1 : (d == -8 ? -1 : 0));  // current rank
                int f = f0 + (d == 1 ? 1 : (d == -1 ? -1 : 0));  // current file
                
                while (t >= 0 && t < 64 && r >= 0 && r < 8 && f >= 0 && f < 8) {
                    if ((own >> t) & 1) break;  // hit own piece
                    push_move(moves, sq, t);
                    if ((occ >> t) & 1) break;  // hit any piece
                    
                    // Move to next square and update rank/file
                    t += d;
                    r += (d == 8 ? 1 : (d == -8 ? -1 : 0));
                    f += (d == 1 ? 1 : (d == -1 ? -1 : 0));
                }
            }
        }
    }

    // Bishops - FIXED
    {
        uint64_t bb = P[W ? WB : BB];
        while (bb) {
            int sq = pop_lsb(bb);
            int r0 = sq / 8;  // starting rank
            int f0 = sq % 8;  // starting file
            
            for (int d : bishop_dirs) {
                int t = sq + d;
                int r = r0 + (d == 9 || d == 7 ? 1 : -1);  // current rank
                int f = f0 + (d == 9 || d == -7 ? 1 : -1);  // current file
                
                while (t >= 0 && t < 64 && r >= 0 && r < 8 && f >= 0 && f < 8) {
                    if ((own >> t) & 1) break;  // hit own piece
                    push_move(moves, sq, t);
                    if ((occ >> t) & 1) break;  // hit any piece
                    
                    // Move to next square and update rank/file
                    t += d;
                    r += (d == 9 || d == 7 ? 1 : -1);
                    f += (d == 9 || d == -7 ? 1 : -1);
                }
            }
        }
    }

    // Queens - FIXED (combines rook and bishop moves)
    {
        uint64_t bb = P[W ? WQ : BQ];
        while (bb) {
            int sq = pop_lsb(bb);
            int r0 = sq / 8;  // starting rank
            int f0 = sq % 8;  // starting file
            
            // Rook-like moves (horizontal/vertical)
            for (int d : {8, -8, 1, -1}) {
                int t = sq + d;
                int r = r0 + (d == 8 ? 1 : (d == -8 ? -1 : 0));
                int f = f0 + (d == 1 ? 1 : (d == -1 ? -1 : 0));
                
                while (t >= 0 && t < 64 && r >= 0 && r < 8 && f >= 0 && f < 8) {
                    if ((own >> t) & 1) break;
                    push_move(moves, sq, t);
                    if ((occ >> t) & 1) break;
                    
                    t += d;
                    r += (d == 8 ? 1 : (d == -8 ? -1 : 0));
                    f += (d == 1 ? 1 : (d == -1 ? -1 : 0));
                }
            }
            
            // Bishop-like moves (diagonal)
            for (int d : {9, 7, -9, -7}) {
                int t = sq + d;
                int r = r0 + (d == 9 || d == 7 ? 1 : -1);  // current rank
                int f = f0 + (d == 9 || d == -7 ? 1 : -1);  // current file
                
                while (t >= 0 && t < 64 && r >= 0 && r < 8 && f >= 0 && f < 8) {
                    if ((own >> t) & 1) break;
                    push_move(moves, sq, t);
                    if ((occ >> t) & 1) break;
                    
                    t += d;
                    r += (d == 9 || d == 7 ? 1 : -1);
                    f += (d == 9 || d == -7 ? 1 : -1);
                }
            }
        }
    }

    // 5) King moves + castling
    {
        uint64_t king = P[W ? WK : BK];
        int sq = pop_lsb(king);
        
        // Helper: is square attacked by opponent? (for normal king moves)
        auto is_attacked = [&](int sq) -> bool {
            // First check pawn attacks
            int pawn_offset1 = W ? -7 : 7;
            int pawn_offset2 = W ? -9 : 9;
            
            // Check if pawn can attack this square
            if (sq + pawn_offset1 >= 0 && sq + pawn_offset1 < 64 && 
                abs((sq & 7) - ((sq + pawn_offset1) & 7)) == 1) {
                if ((1ULL << (sq + pawn_offset1)) & P[W ? BP : WP]) return true;
            }
            
            if (sq + pawn_offset2 >= 0 && sq + pawn_offset2 < 64 &&
                abs((sq & 7) - ((sq + pawn_offset2) & 7)) == 1) {
                if ((1ULL << (sq + pawn_offset2)) & P[W ? BP : WP]) return true;
            }
            
            // Check knight attacks
            uint64_t knights = P[W ? BN : WN];
            uint64_t knight_attacks = [&](int s) {
                uint64_t b = 1ULL << s;
                uint64_t l1 = (b >> 1) & 0x7f7f7f7f7f7f7f7fULL;
                uint64_t l2 = (b >> 2) & 0x3f3f3f3f3f3f3f3fULL;
                uint64_t r1 = (b << 1) & 0xfefefefefefefefeULL;
                uint64_t r2 = (b << 2) & 0xfcfcfcfcfcfcfcfcULL;
                uint64_t h1 = l1 | r1;
                uint64_t h2 = l2 | r2;
                return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
            }(sq);
            
            if (knight_attacks & knights) return true;
            
            // Check sliding pieces (rooks, bishops, queens)
            // Rook-like moves (horizontal/vertical)
            for (int d : {+8, -8, +1, -1}) {
                int t = sq + d;
                int f0 = sq & 7;
                while (t >= 0 && t < 64 && abs((t & 7) - f0) <= (d == +8 || d == -8 ? 0 : 1)) {
                    uint64_t bit = 1ULL << t;
                    if (bit & occ) {
                        if (bit & (P[W ? BR : WR] | P[W ? BQ : WQ])) return true;
                        break;
                    }
                    t += d;
                }
            }
            
            // Bishop-like moves (diagonal)
            for (int d : {+9, +7, -9, -7}) {
                int t = sq + d;
                int f0 = sq & 7;
                while (t >= 0 && t < 64 && abs((t & 7) - f0) == 1) {
                    uint64_t bit = 1ULL << t;
                    if (bit & occ) {
                        if (bit & (P[W ? BB : WB] | P[W ? BQ : WQ])) return true;
                        break;
                    }
                    t += d;
                }
            }
            
            // Check king proximity (for adjacent kings)
            uint64_t opp_king = P[W ? BK : WK];
            if (opp_king) {
                unsigned long idx;
                _BitScanForward64(&idx, opp_king);
                int kingsq = static_cast<int>(idx);
                int dx = abs((kingsq & 7) - (sq & 7));
                int dy = abs((kingsq >> 3) - (sq >> 3));
                if (dx <= 1 && dy <= 1) return true;
            }
            
            return false;
        };
        
        // one‑square moves - WITH CHECK VALIDATION
        for (int d : KingD) {
            int t = sq + d;
            if (t < 0 || t >= 64) continue;
            if (abs((t & 7) - (sq & 7)) > 1) continue;
            if (!(own & (1ULL << t))) {
                if (!is_attacked(t)) {
                    push_move(moves, sq, t);
                }
            }
        }
        
        // castling
        // Helper: is square attacked by opponent? (for castling, uses pseudo-legal moves)
        auto is_attacked_castle = [&](int sq) -> bool {
            Board tmp = b;
            tmp.set_white_to_move(!W);  // Flip turn
            auto opp_moves = generate_pseudo_legal_moves(tmp); // avoid calling generate_moves()
            for (const auto& m : opp_moves) {
                int to = (m[2] - 'a') + (m[3] - '1') * 8;
                if (to == sq) return true;
            }
            return false;
        };

        int rights = b.castling_rights();
        if (W) {
            // White
            if ((rights & 1) && !(occ & ((1ULL<<5)|(1ULL<<6)))) {
                // e1, f1, g1 must not be attacked
                if (!is_attacked_castle(4) && !is_attacked_castle(5) && !is_attacked_castle(6))
                    moves.push_back("e1g1");
            }
            if ((rights & 2) && !(occ & ((1ULL<<1)|(1ULL<<2)|(1ULL<<3)))) {
                // e1, d1, c1 must not be attacked
                if (!is_attacked_castle(4) && !is_attacked_castle(3) && !is_attacked_castle(2))
                    moves.push_back("e1c1");
            }
        } else {
            // Black
            if ((rights & 4) && !(occ & ((1ULL<<61)|(1ULL<<62)))) {
                // e8, f8, g8 must not be attacked
                if (!is_attacked_castle(60) && !is_attacked_castle(61) && !is_attacked_castle(62))
                    moves.push_back("e8g8");
            }
            if ((rights & 8) && !(occ & ((1ULL<<57)|(1ULL<<58)|(1ULL<<59)))) {
                // e8, d8, c8 must not be attacked
                if (!is_attacked_castle(60) && !is_attacked_castle(59) && !is_attacked_castle(58))
                    moves.push_back("e8c8");
            }
        }
    }

    // Remove moves that land on own king or capture opponent king
    int own_king_sq = -1, opp_king_sq = -1;
    {
        const uint64_t *P = b.pieces();
        uint64_t own_king_bb = P[W ? WK : BK];
        uint64_t opp_king_bb = P[W ? BK : WK];
        if (own_king_bb) {
            unsigned long idx;
            _BitScanForward64(&idx, own_king_bb);
            own_king_sq = static_cast<int>(idx);
        }
        if (opp_king_bb) {
            unsigned long idx;
            _BitScanForward64(&idx, opp_king_bb);
            opp_king_sq = static_cast<int>(idx);
        }
    }
    std::vector<std::string> filtered;
    for (const auto& m : moves) {
        int to = (m[2]-'a') + (m[3]-'1')*8;
        // Do not allow moves that land on own king or capture opponent king
        if (to != own_king_sq && to != opp_king_sq)
            filtered.push_back(m);
    }
    moves.swap(filtered);

    // --- ENFORCE: If king is in check, only allow moves that get out of check ---
    if (b.in_check(W)) {
        std::vector<std::string> legal;
        for (const auto& m : moves) {
            Board tmp = b;
            try {
                tmp.make_move(m);
                if (!tmp.in_check(W)) {
                    legal.push_back(m);
                }
            } catch (...) {
                // Ignore illegal moves
            }
        }
        moves.swap(legal);
    }

    return moves;
}
