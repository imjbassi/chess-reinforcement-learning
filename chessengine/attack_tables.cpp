#include <cstdint>
#include "attack_tables.h"

// Knight attacks lookup
uint64_t knight_attacks(int square) {
    uint64_t b = 1ULL << square;
    uint64_t l1 = (b >> 1) & 0x7f7f7f7f7f7f7f7fULL;
    uint64_t l2 = (b >> 2) & 0x3f3f3f3f3f3f3f3fULL;
    uint64_t r1 = (b << 1) & 0xfefefefefefefefeULL;
    uint64_t r2 = (b << 2) & 0xfcfcfcfcfcfcfcfcULL;
    uint64_t h1 = l1 | r1;
    uint64_t h2 = l2 | r2;
    return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
}

// Helper for sliding pieces
uint64_t sliding_attacks(int square, uint64_t occupied, const int* directions, int num_dirs) {
    uint64_t attacks = 0;
    int rank = square / 8;
    int file = square % 8;
    
    for (int i = 0; i < num_dirs; i++) {
        int dir = directions[i];
        int dr = (dir >= 7 && dir <= 9) ? 1 : (dir >= -9 && dir <= -7) ? -1 : 0;
        int df = (dir == 1 || dir == 9 || dir == -7) ? 1 : (dir == -1 || dir == -9 || dir == 7) ? -1 : 0;
        
        int r = rank + dr;
        int f = file + df;
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            int sq = r * 8 + f;
            uint64_t sq_bb = 1ULL << sq;
            attacks |= sq_bb;
            
            // Stop if we hit an occupied square
            if (occupied & sq_bb) break;
            
            r += dr;
            f += df;
        }
    }
    
    return attacks;
}

// Rook attacks (horizontal/vertical)
uint64_t rook_attacks(int square, uint64_t occupancy) {
    const int rook_dirs[] = {8, -8, 1, -1};  // N, S, E, W
    return sliding_attacks(square, occupancy, rook_dirs, 4);
}

// Bishop attacks (diagonal)
uint64_t bishop_attacks(int square, uint64_t occupancy) {
    const int bishop_dirs[] = {9, 7, -7, -9};  // NE, NW, SE, SW
    return sliding_attacks(square, occupancy, bishop_dirs, 4);
}

// Queen attacks (rook + bishop)
uint64_t queen_attacks(int square, uint64_t occupancy) {
    return rook_attacks(square, occupancy) | bishop_attacks(square, occupancy);
}

// King attacks
uint64_t king_attacks(int square) {
    uint64_t b = 1ULL << square;
    uint64_t attacks = ((b << 1) & 0xfefefefefefefefeULL) | ((b >> 1) & 0x7f7f7f7f7f7f7f7fULL);
    b |= attacks;
    attacks |= ((b << 8) | (b >> 8));
    return attacks & ~(1ULL << square);
}

// Pawn attacks
uint64_t pawn_attacks(int square, bool white) {
    uint64_t b = 1ULL << square;
    if (white) {
        return (((b << 7) & 0x7f7f7f7f7f7f7f7fULL) | ((b << 9) & 0xfefefefefefefefeULL));
    } else {
        return (((b >> 7) & 0xfefefefefefefefeULL) | ((b >> 9) & 0x7f7f7f7f7f7f7f7fULL));
    }
}