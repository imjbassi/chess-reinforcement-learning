// chessengine/bitops.h
// Portable bit-scan helpers (MSVC intrinsics on Windows, GCC/Clang builtins elsewhere).
#pragma once
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
static inline int lsb_index(uint64_t bb) {
    unsigned long idx;
    _BitScanForward64(&idx, bb);
    return static_cast<int>(idx);
}
#else
static inline int lsb_index(uint64_t bb) {
    return __builtin_ctzll(bb);
}
#endif
