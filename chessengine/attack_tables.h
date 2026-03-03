// chessengine/attack_tables.h
#pragma once

uint64_t knight_attacks(int square);
uint64_t rook_attacks(int square, uint64_t occupancy);
uint64_t bishop_attacks(int square, uint64_t occupancy);
uint64_t queen_attacks(int square, uint64_t occupancy);
uint64_t king_attacks(int square);
uint64_t pawn_attacks(int square, bool white);
