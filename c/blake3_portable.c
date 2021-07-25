// This code is a translation from the SSE41 code to portable __builtins.
// Consequently, it allows use of SIMD on Apple M1 processors and should
// work on future SIMD extensions such as those for RISC V.
//
// If this code is compiled for a processor that lacks SIMD instructions,
// the compiler will substitute the appropriate single-width code.
//
// We avoid using __builtin_shuffle and __builtin_shufflevector because
// gcc and clang disagree over those.  In most cases, the workaround is
// to use array initialization, and the compiler usually generates the
// appropriate shuffle instruction anyway.

#include "blake3_impl.h"
#include <string.h>

#define DEGREE 4

typedef uint32_t v4ui __attribute__ ((vector_size (16)));
typedef uint8_t v16ub __attribute__ ((vector_size (16)));

INLINE v4ui loadu(const uint8_t src[16]) {
  v4ui x;
  memcpy(&x, src, 16);
  return x;
}

INLINE void storeu(v4ui src, uint8_t dest[16]) {
  memcpy(dest, &src, 16);
}

INLINE v4ui set1(uint32_t x) {
  const v4ui val = {x, x, x, x};
  return val;
}

INLINE v4ui set4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  const v4ui val = {a, b, c, d};
  return val;
}

INLINE v4ui rot16(v4ui x) {
  v16ub b;
  storeu(x, (uint8_t *)&b);
  v16ub bs = {
    b[ 2], b[ 3], b[ 0], b[ 1], b[ 6], b[ 7], b[ 4], b[ 5],
    b[10], b[11], b[ 8], b[ 9], b[14], b[15], b[12], b[13]
  };
  return loadu((uint8_t *)&bs);
}

INLINE v4ui rot8(v4ui x) {
  v16ub b;
  storeu(x, (uint8_t *)&b);
  v16ub bs = {
    b[ 1], b[ 2], b[ 3], b[ 0], b[ 5], b[ 6], b[ 7], b[ 4],
    b[ 9], b[10], b[11], b[ 8], b[13], b[14], b[15], b[12]
  };
  return loadu((uint8_t *)&bs);
}

INLINE v4ui rot(v4ui x, uint32_t c)
{
  switch (c) {
  case 16:
    return rot16(x);
  case 8:
    return rot8(x);
  default:
    return (x >> c) ^ (x << (32 - c));
  }
}

INLINE void g1(v4ui *row0, v4ui *row1, v4ui *row2, v4ui *row3, v4ui m) {
  *row0 += m + *row1;
  *row3 ^= *row0;
  *row3 = rot(*row3, 16);
  *row2 += *row3;
  *row1 ^= *row2;
  *row1 = rot(*row1, 12);
}

INLINE void g2(v4ui *row0, v4ui *row1, v4ui *row2, v4ui *row3, v4ui m) {
  *row0 += m + *row1;
  *row3 ^= *row0;
  *row3 = rot(*row3, 8);
  *row2 += *row3;
  *row1 ^= *row2;
  *row1 = rot(*row1, 7);
}

// Note the optimization here of leaving row1 as the unrotated row, rather than
// row0. All the message loads below are adjusted to compensate for this. See
// discussion at https://github.com/sneves/blake2-avx2/pull/4
INLINE void diagonalize(v4ui *row0, v4ui *row2, v4ui *row3) {
  const v4ui r0 = *row0;
  const v4ui r0s = {r0[3], r0[0], r0[1], r0[2]};
  *row0 = r0s;
  const v4ui r3 = *row3;
  const v4ui r3s = {r3[2], r3[3], r3[0], r3[1]};
  *row3 = r3s;
  const v4ui r2 = *row2;
  const v4ui r2s = {r2[1], r2[2], r2[3], r2[0]};
  *row2 = r2s;
}

INLINE void undiagonalize(v4ui *row0, v4ui *row2, v4ui *row3) {
  const v4ui r0 = *row0;
  const v4ui r0s = {r0[1], r0[2], r0[3], r0[0]};
  *row0 = r0s;
  const v4ui r3 = *row3;
  const v4ui r3s = {r3[2], r3[3], r3[0], r3[1]};
  *row3 = r3s;
  const v4ui r2 = *row2;
  const v4ui r2s = {r2[3], r2[0], r2[1], r2[2]};
  *row2 = r2s;
}

INLINE void fixed_round(v4ui rows[4],
                        v4ui *pm0, v4ui *pm1, v4ui *pm2, v4ui *pm3) {
  const v4ui m0 = *pm0;
  const v4ui m1 = *pm1;
  const v4ui m2 = *pm2;
  const v4ui m3 = *pm3;
  const v4ui t0a = {m0[2], m0[1], m1[1], m1[3]};
  const v4ui t0 = {t0a[1], t0a[2], t0a[3], t0a[0]};
  g1(&rows[0], &rows[1], &rows[2], &rows[3], t0);
  const v4ui t1a = {m2[2], m2[2], m3[3], m3[3]};
  const v4ui tt1 = {m0[3], m0[3], m0[0], m0[0]};
  const v4ui t1 = {tt1[0], t1a[1], tt1[2], t1a[3]};
  g2(&rows[0], &rows[1], &rows[2], &rows[3], t1);
  diagonalize(&rows[0], &rows[2], &rows[3]);
  const v4ui t2a = {m3[0], m3[1], m1[0], m1[1]};
  const v4ui tt2 = {t2a[0], t2a[1], t2a[2], m2[3]};
  const v4ui t2 = {tt2[0], tt2[2], tt2[3], tt2[1]};
  g1(&rows[0], &rows[1], &rows[2], &rows[3], t2);
  const v4ui t3a = {m1[2], m3[2], m1[3], m3[3]};
  const v4ui tt3 = {m2[0], t3a[0], m2[1], t3a[1]};
  const v4ui t3 = {tt3[2], tt3[3], tt3[1], tt3[0]};
  g2(&rows[0], &rows[1], &rows[2], &rows[3], t3);
  undiagonalize(&rows[0], &rows[2], &rows[3]);
  *pm0 = t0;
  *pm1 = t1;
  *pm2 = t2;
  *pm3 = t3;
}

INLINE void compress_pre(v4ui rows[4], const uint32_t cv[8],
                         const uint8_t block[BLAKE3_BLOCK_LEN],
                         uint8_t block_len, uint64_t counter, uint8_t flags) {
  rows[0] = loadu((uint8_t *)&cv[0]);
  rows[1] = loadu((uint8_t *)&cv[4]);
  rows[2] = set4(IV[0], IV[1], IV[2], IV[3]);
  rows[3] = set4(counter_low(counter), counter_high(counter),
                 (uint32_t)block_len, (uint32_t)flags);

  v4ui m0 = loadu(&block[sizeof(v4ui) * 0]);
  v4ui m1 = loadu(&block[sizeof(v4ui) * 1]);
  v4ui m2 = loadu(&block[sizeof(v4ui) * 2]);
  v4ui m3 = loadu(&block[sizeof(v4ui) * 3]);

  // Round 1. The first round permutes the message words from the original
  // input order, into the groups that get mixed in parallel.
  const v4ui t0 = {m0[0], m0[2], m1[0], m1[2]};     //  0  2  4  6
  g1(&rows[0], &rows[1], &rows[2], &rows[3], t0);
  const v4ui t1 = {m0[1], m0[3], m1[1], m1[3]};     //  1  3  5  7
  g2(&rows[0], &rows[1], &rows[2], &rows[3], t1);
  diagonalize(&rows[0], &rows[2], &rows[3]);
  const v4ui t2a = {m2[0], m2[2], m3[0], m3[2]};    //  8 10 12 14
  const v4ui t2 = {t2a[3], t2a[0], t2a[1], t2a[2]}; // 14  8 10 12
  g1(&rows[0], &rows[1], &rows[2], &rows[3], t2);
  const v4ui t3a = {m2[1], m2[3], m3[1], m3[3]};    //  9 11 13 15
  const v4ui t3 = {t3a[3], t3a[0], t3a[1], t3a[2]}; // 15  9 11 13
  g2(&rows[0], &rows[1], &rows[2], &rows[3], t3);
  undiagonalize(&rows[0], &rows[2], &rows[3]);
  m0 = t0;
  m1 = t1;
  m2 = t2;
  m3 = t3;

  // Round 2 and all following rounds apply a fixed permutation
  // to the message words from the round before.
  for (unsigned round = 2; round <= 7; ++round)
    fixed_round(rows, &m0, &m1, &m2, &m3);
}

void blake3_compress_in_place_portable(uint32_t cv[8],
                                       const uint8_t block[BLAKE3_BLOCK_LEN],
                                       uint8_t block_len, uint64_t counter,
                                       uint8_t flags) {
  v4ui rows[4];
  compress_pre(rows, cv, block, block_len, counter, flags);
  storeu(rows[0] ^ rows[2], (uint8_t *)&cv[0]);
  storeu(rows[1] ^ rows[3], (uint8_t *)&cv[4]);
}

void blake3_compress_xof_portable(const uint32_t cv[8],
                                  const uint8_t block[BLAKE3_BLOCK_LEN],
                                  uint8_t block_len, uint64_t counter,
                                  uint8_t flags, uint8_t out[64]) {
  v4ui rows[4];
  compress_pre(rows, cv, block, block_len, counter, flags);
  storeu(rows[0] ^ rows[2], &out[0]);
  storeu(rows[1] ^ rows[3], &out[16]);
  storeu(rows[2] ^ loadu((uint8_t *)&cv[0]), &out[32]);
  storeu(rows[3] ^ loadu((uint8_t *)&cv[4]), &out[48]);
}

INLINE void round_fn(v4ui v[16], v4ui m[16], size_t r) {
  v[0] += m[(size_t)MSG_SCHEDULE[r][0]];
  v[1] += m[(size_t)MSG_SCHEDULE[r][2]];
  v[2] += m[(size_t)MSG_SCHEDULE[r][4]];
  v[3] += m[(size_t)MSG_SCHEDULE[r][6]];
  v[0] += v[4];
  v[1] += v[5];
  v[2] += v[6];
  v[3] += v[7];
  v[12] ^= v[0];
  v[13] ^= v[1];
  v[14] ^= v[2];
  v[15] ^= v[3];
  v[12] = rot(v[12], 16);
  v[13] = rot(v[13], 16);
  v[14] = rot(v[14], 16);
  v[15] = rot(v[15], 16);
  v[8] += v[12];
  v[9] += v[13];
  v[10] += v[14];
  v[11] += v[15];
  v[4] ^= v[8];
  v[5] ^= v[9];
  v[6] ^= v[10];
  v[7] ^= v[11];
  v[4] = rot(v[4], 12);
  v[5] = rot(v[5], 12);
  v[6] = rot(v[6], 12);
  v[7] = rot(v[7], 12);
  v[0] += m[(size_t)MSG_SCHEDULE[r][1]];
  v[1] += m[(size_t)MSG_SCHEDULE[r][3]];
  v[2] += m[(size_t)MSG_SCHEDULE[r][5]];
  v[3] += m[(size_t)MSG_SCHEDULE[r][7]];
  v[0] += v[4];
  v[1] += v[5];
  v[2] += v[6];
  v[3] += v[7];
  v[12] ^= v[0];
  v[13] ^= v[1];
  v[14] ^= v[2];
  v[15] ^= v[3];
  v[12] = rot(v[12], 8);
  v[13] = rot(v[13], 8);
  v[14] = rot(v[14], 8);
  v[15] = rot(v[15], 8);
  v[8] += v[12];
  v[9] += v[13];
  v[10] += v[14];
  v[11] += v[15];
  v[4] ^= v[8];
  v[5] ^= v[9];
  v[6] ^= v[10];
  v[7] ^= v[11];
  v[4] = rot(v[4], 7);
  v[5] = rot(v[5], 7);
  v[6] = rot(v[6], 7);
  v[7] = rot(v[7], 7);

  v[0] += m[(size_t)MSG_SCHEDULE[r][8]];
  v[1] += m[(size_t)MSG_SCHEDULE[r][10]];
  v[2] += m[(size_t)MSG_SCHEDULE[r][12]];
  v[3] += m[(size_t)MSG_SCHEDULE[r][14]];
  v[0] += v[5];
  v[1] += v[6];
  v[2] += v[7];
  v[3] += v[4];
  v[15] ^= v[0];
  v[12] ^= v[1];
  v[13] ^= v[2];
  v[14] ^= v[3];
  v[15] = rot(v[15], 16);
  v[12] = rot(v[12], 16);
  v[13] = rot(v[13], 16);
  v[14] = rot(v[14], 16);
  v[10] += v[15];
  v[11] += v[12];
  v[8] += v[13];
  v[9] += v[14];
  v[5] ^= v[10];
  v[6] ^= v[11];
  v[7] ^= v[8];
  v[4] ^= v[9];
  v[5] = rot(v[5], 12);
  v[6] = rot(v[6], 12);
  v[7] = rot(v[7], 12);
  v[4] = rot(v[4], 12);
  v[0] += m[(size_t)MSG_SCHEDULE[r][9]];
  v[1] += m[(size_t)MSG_SCHEDULE[r][11]];
  v[2] += m[(size_t)MSG_SCHEDULE[r][13]];
  v[3] += m[(size_t)MSG_SCHEDULE[r][15]];
  v[0] += v[5];
  v[1] += v[6];
  v[2] += v[7];
  v[3] += v[4];
  v[15] ^= v[0];
  v[12] ^= v[1];
  v[13] ^= v[2];
  v[14] ^= v[3];
  v[15] = rot(v[15], 8);
  v[12] = rot(v[12], 8);
  v[13] = rot(v[13], 8);
  v[14] = rot(v[14], 8);
  v[10] += v[15];
  v[11] += v[12];
  v[8] += v[13];
  v[9] += v[14];
  v[5] ^= v[10];
  v[6] ^= v[11];
  v[7] ^= v[8];
  v[4] ^= v[9];
  v[5] = rot(v[5], 7);
  v[6] = rot(v[6], 7);
  v[7] = rot(v[7], 7);
  v[4] = rot(v[4], 7);
}

INLINE void transpose_vecs(v4ui vecs[DEGREE]) {
  // Interleave 32-bit lanes. The low unpack is lanes 00/11 and the high is
  // 22/33. Note that this doesn't split the vector into two lanes, as the
  // AVX2 counterparts do.
  const v4ui ab_01 = {vecs[0][0], vecs[1][0], vecs[0][1], vecs[1][1]};
  const v4ui ab_23 = {vecs[0][2], vecs[1][2], vecs[0][3], vecs[1][3]};
  const v4ui cd_01 = {vecs[2][0], vecs[3][0], vecs[2][1], vecs[3][1]};
  const v4ui cd_23 = {vecs[2][2], vecs[3][2], vecs[2][3], vecs[3][3]};

  // Interleave 64-bit lanes.
  const v4ui abcd_0 = {ab_01[0], ab_01[1], cd_01[0], cd_01[1]};
  const v4ui abcd_1 = {ab_01[2], ab_01[3], cd_01[2], cd_01[3]};
  const v4ui abcd_2 = {ab_23[0], ab_23[1], cd_23[0], cd_23[1]};
  const v4ui abcd_3 = {ab_23[2], ab_23[3], cd_23[2], cd_23[3]};

  vecs[0] = abcd_0;
  vecs[1] = abcd_1;
  vecs[2] = abcd_2;
  vecs[3] = abcd_3;
}

INLINE void transpose_msg_vecs(const uint8_t *const *inputs,
                               size_t block_offset, v4ui out[16]) {
  out[0] = loadu(&inputs[0][block_offset + 0 * sizeof(v4ui)]);
  out[1] = loadu(&inputs[1][block_offset + 0 * sizeof(v4ui)]);
  out[2] = loadu(&inputs[2][block_offset + 0 * sizeof(v4ui)]);
  out[3] = loadu(&inputs[3][block_offset + 0 * sizeof(v4ui)]);
  out[4] = loadu(&inputs[0][block_offset + 1 * sizeof(v4ui)]);
  out[5] = loadu(&inputs[1][block_offset + 1 * sizeof(v4ui)]);
  out[6] = loadu(&inputs[2][block_offset + 1 * sizeof(v4ui)]);
  out[7] = loadu(&inputs[3][block_offset + 1 * sizeof(v4ui)]);
  out[8] = loadu(&inputs[0][block_offset + 2 * sizeof(v4ui)]);
  out[9] = loadu(&inputs[1][block_offset + 2 * sizeof(v4ui)]);
  out[10] = loadu(&inputs[2][block_offset + 2 * sizeof(v4ui)]);
  out[11] = loadu(&inputs[3][block_offset + 2 * sizeof(v4ui)]);
  out[12] = loadu(&inputs[0][block_offset + 3 * sizeof(v4ui)]);
  out[13] = loadu(&inputs[1][block_offset + 3 * sizeof(v4ui)]);
  out[14] = loadu(&inputs[2][block_offset + 3 * sizeof(v4ui)]);
  out[15] = loadu(&inputs[3][block_offset + 3 * sizeof(v4ui)]);
  for (size_t i = 0; i < 4; ++i) {
    __builtin_prefetch(&inputs[i][block_offset + 256]);
  }
  transpose_vecs(&out[0]);
  transpose_vecs(&out[4]);
  transpose_vecs(&out[8]);
  transpose_vecs(&out[12]);
}

INLINE void load_counters(uint64_t counter, bool increment_counter,
                          v4ui *out_lo, v4ui *out_hi) {
  const v4ui mask = set1(-(int32_t)increment_counter);
  const v4ui add0 = {0, 1, 2, 3};
  const v4ui add1 = mask & add0;
  v4ui l = (uint32_t)counter + add1;
  v4ui negcarry = add1 > l;
  v4ui h = (uint32_t)(counter >> 32) - negcarry;
  *out_lo = l;
  *out_hi = h;
}

void blake3_hash4_portable(const uint8_t *const *inputs, size_t blocks,
                           const uint32_t key[8], uint64_t counter,
                           bool increment_counter, uint8_t flags,
                           uint8_t flags_start, uint8_t flags_end,
                           uint8_t *out) {
  v4ui h_vecs[8] = {
      set1(key[0]), set1(key[1]), set1(key[2]), set1(key[3]),
      set1(key[4]), set1(key[5]), set1(key[6]), set1(key[7]),
  };
  v4ui counter_low_vec, counter_high_vec;
  load_counters(counter, increment_counter, &counter_low_vec,
                &counter_high_vec);
  uint8_t block_flags = flags | flags_start;

  for (size_t block = 0; block < blocks; block++) {
    if (block + 1 == blocks) {
      block_flags |= flags_end;
    }
    v4ui block_len_vec = set1(BLAKE3_BLOCK_LEN);
    v4ui block_flags_vec = set1(block_flags);
    v4ui msg_vecs[16];
    transpose_msg_vecs(inputs, block * BLAKE3_BLOCK_LEN, msg_vecs);

    v4ui v[16] = {
        h_vecs[0],       h_vecs[1],        h_vecs[2],     h_vecs[3],
        h_vecs[4],       h_vecs[5],        h_vecs[6],     h_vecs[7],
        set1(IV[0]),     set1(IV[1]),      set1(IV[2]),   set1(IV[3]),
        counter_low_vec, counter_high_vec, block_len_vec, block_flags_vec,
    };
    round_fn(v, msg_vecs, 0);
    round_fn(v, msg_vecs, 1);
    round_fn(v, msg_vecs, 2);
    round_fn(v, msg_vecs, 3);
    round_fn(v, msg_vecs, 4);
    round_fn(v, msg_vecs, 5);
    round_fn(v, msg_vecs, 6);
    h_vecs[0] = v[0] ^ v[8];
    h_vecs[1] = v[1] ^ v[9];
    h_vecs[2] = v[2] ^ v[10];
    h_vecs[3] = v[3] ^ v[11];
    h_vecs[4] = v[4] ^ v[12];
    h_vecs[5] = v[5] ^ v[13];
    h_vecs[6] = v[6] ^ v[14];
    h_vecs[7] = v[7] ^ v[15];

    block_flags = flags;
  }

  transpose_vecs(&h_vecs[0]);
  transpose_vecs(&h_vecs[4]);
  // The first four vecs now contain the first half of each output, and the
  // second four vecs contain the second half of each output.
  storeu(h_vecs[0], &out[0 * sizeof(v4ui)]);
  storeu(h_vecs[4], &out[1 * sizeof(v4ui)]);
  storeu(h_vecs[1], &out[2 * sizeof(v4ui)]);
  storeu(h_vecs[5], &out[3 * sizeof(v4ui)]);
  storeu(h_vecs[2], &out[4 * sizeof(v4ui)]);
  storeu(h_vecs[6], &out[5 * sizeof(v4ui)]);
  storeu(h_vecs[3], &out[6 * sizeof(v4ui)]);
  storeu(h_vecs[7], &out[7 * sizeof(v4ui)]);
}

INLINE void hash_one_portable(const uint8_t *input, size_t blocks,
                              const uint32_t key[8], uint64_t counter,
                              uint8_t flags, uint8_t flags_start,
                              uint8_t flags_end, uint8_t out[BLAKE3_OUT_LEN]) {
  uint32_t cv[8];
  memcpy(cv, key, BLAKE3_KEY_LEN);
  uint8_t block_flags = flags | flags_start;
  while (blocks > 0) {
    if (blocks == 1) {
      block_flags |= flags_end;
    }
    blake3_compress_in_place_portable(cv, input, BLAKE3_BLOCK_LEN, counter,
                                      block_flags);
    input = &input[BLAKE3_BLOCK_LEN];
    blocks -= 1;
    block_flags = flags;
  }
  memcpy(out, cv, BLAKE3_OUT_LEN);
}

void blake3_hash_many_portable(const uint8_t *const *inputs, size_t num_inputs,
                            size_t blocks, const uint32_t key[8],
                            uint64_t counter, bool increment_counter,
                            uint8_t flags, uint8_t flags_start,
                            uint8_t flags_end, uint8_t *out) {
  while (num_inputs >= DEGREE) {
    blake3_hash4_portable(inputs, blocks, key, counter, increment_counter,
                          flags, flags_start, flags_end, out);
    if (increment_counter) {
      counter += DEGREE;
    }
    inputs += DEGREE;
    num_inputs -= DEGREE;
    out = &out[DEGREE * BLAKE3_OUT_LEN];
  }
  while (num_inputs > 0) {
    hash_one_portable(inputs[0], blocks, key, counter, flags, flags_start,
                      flags_end, out);
    if (increment_counter) {
      counter += 1;
    }
    inputs += 1;
    num_inputs -= 1;
    out = &out[BLAKE3_OUT_LEN];
  }
}
