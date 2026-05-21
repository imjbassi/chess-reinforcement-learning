```cpp
#include <cstdint>
#include "attack_tables.h"

// Knight attacks lookup table
// Knights move in an L-shape: 2 squares in one direction and 1 square perpendicular
uint64_t knight_attacks(int square) {
    const uint64_t b = 1ULL << square;
    const uint64_t l1 = (b >> 1) & 0x7f7f7f7f7f7f7f7fULL;  // Left 1 file
    const uint64_t l2 = (b >> 2) & 0x3f3f3f3f3f3f3f3fULL;  // Left 2 files
    const uint64_t r1 = (b << 1) & 0xfefefefefefefefeULL;  // Right 1 file
    const uint64_t r2 = (b << 2) & 0xfcfcfcfcfcfcfcfcULL;  // Right 2 files
    const uint64_t h1 = l1 | r1;  // Horizontal 1
    const uint64_t h2 = l2 | r2;  // Horizontal 2
    return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
}

// Helper function for sliding pieces (rooks, bishops, queens)
// Generates attacks along specified directions until blocked by occupied squares
uint64_t sliding_attacks(int square, uint64_t occupied, const int* directions, int num_dirs) {
    uint64_t attacks = 0;
    const int rank = square / 8;
    const int file = square % 8;
    
    for (int i = 0; i < num_dirs; i++) {
        const int dir = directions[i];
        
        // Determine rank and file deltas from direction
        int dr = 0;
        int df = 0;
        
        switch (dir) {
            case 8:   dr = 1;  df = 0;  break;  // North
            case -8:  dr = -1; df = 0;  break;  // South
            case 1:   dr = 0;  df = 1;  break;  // East
            case -1:  dr = 0;  df = -1; break;  // West
            case 9:   dr = 1;  df = 1;  break;  // Northeast
            case 7:   dr = 1;  df = -1; break;  // Northwest
            case -7:  dr = -1; df = 1;  break;  // Southeast
            case -9:  dr = -1; df = -1; break;  // Southwest
            default:  break;
        }
        
        int r = rank + dr;
        int f = file + df;
        
        // Slide along the direction until hitting board edge or occupied square
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            const int sq = r * 8 + f;
            const uint64_t sq_bb = 1ULL << sq;
            attacks |= sq_bb;
            
            // Stop if we hit an occupied square (include capture square)
            if (occupied & sq_bb) {
                break;
            }
            
            r += dr;
            f += df;
        }
    }
    
    return attacks;
}

// Rook attacks (horizontal and vertical directions)
uint64_t rook_attacks(int square, uint64_t occupancy) {
    static const int rook_dirs[] = {8, -8, 1, -1};  // North, South, East, West
    return sliding_attacks(square, occupancy, rook_dirs, 4);
}

// Bishop attacks (diagonal directions)
uint64_t bishop_attacks(int square, uint64_t occupancy) {
    static const int bishop_dirs[] = {9, 7, -7, -9};  // NE, NW, SE, SW
    return sliding_attacks(square, occupancy, bishop_dirs, 4);
}

// Queen attacks (combination of rook and bishop)
uint64_t queen_attacks(int square, uint64_t occupancy) {
    return rook_attacks(square, occupancy) | bishop_attacks(square, occupancy);
}

// King attacks (one square in any direction)
uint64_t king_attacks(int square) {
    const uint64_t b = 1ULL << square;
    const uint64_t left = (b >> 1) & 0x7f7f7f7f7f7f7f7fULL;
    const uint64_t right = (b << 1) & 0xfefefefefefefefeULL;
    const uint64_t horizontal = left | right;
    const uint64_t extended = b | horizontal;
    return ((extended << 8) | (extended >> 8) | horizontal) & ~b;
}

// Pawn attacks (diagonal captures only, not forward moves)
uint64_t pawn_attacks(int square, bool white) {
    const uint64_t b = 1ULL << square;
    if (white) {
        // White pawns attack diagonally upward (northeast and northwest)
        const uint64_t left_attack = (b << 7) & 0x7f7f7f7f7f7f7f7fULL;   // Northwest (avoid A-file wrap)
        const uint64_t right_attack = (b << 9) & 0xfefefefefefefefeULL;  // Northeast (avoid H-file wrap)
        return left_attack | right_attack;
    } else {
        // Black pawns attack diagonally downward (southeast and southwest)
        const uint64_t left_attack = (b >> 9) & 0x7f7f7f7f7f7f7f7fULL;   // Southwest (avoid A-file wrap)
        const uint64_t right_attack = (b >> 7) & 0xfefefefefefefefeULL;  // Southeast (avoid H-file wrap)
        return left_attack | right_attack;
    }
}
```