// See LICENSE for license details.

// Leave PRINT_TILE at the Makefile's 0. Setting it to 1 makes
// tiled_matmul_auto print its chosen tiling and then exit(), so the matmul
// never runs. It reports (5,6,16) here, which candidates[0] reproduces.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define ACTIVATION NO_ACTIVATION

// NO_BIAS=1 makes tiled_matmul_outer substitute a dummy D pointer of 1,
// which spike's gemmini extension aborts on. Match tiled_matmul_ws_perf.
#define NO_BIAS 0
#define REPEATING_BIAS 1

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

#define CHECK_RESULTS 1

#define WARMUP_TILE_I 5
#define WARMUP_TILE_J 6
#define WARMUP_TILE_K 16

#define MAT_DIM_I 128
#define MAT_DIM_K 256
#define MAT_DIM_J 256

#define A_STRIDE MAT_DIM_K
#define B_STRIDE MAT_DIM_J

static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
static acc_t  full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

static const size_t candidates[][3] = {
    {5,  6, 16},
    {4,  8, 16},
    {8,  4, 16},
    {4,  4, 16},
    {2, 16, 16},
    {6,  5, 16},
};

#define NUM_CANDIDATES (sizeof(candidates) / sizeof(candidates[0]))

static void setup_counters() {
    counter_configure(0, RDMA_BYTES_REC);
    counter_configure(1, WDMA_BYTES_SENT);
    counter_configure(2, EXE_ACTIVE_CYCLE);
    counter_configure(3, LOAD_DMA_WAIT_CYCLE);
    counter_configure(4, STORE_DMA_WAIT_CYCLE);
    counter_configure(5, SCRATCHPAD_A_WAIT_CYCLE);
    counter_configure(6, SCRATCHPAD_B_WAIT_CYCLE);
    counter_configure(7, RESERVATION_STATION_FULL_CYCLES);
}

static void report(uint64_t cycles) {
    const uint64_t total_macs = (uint64_t)MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
    const uint64_t ideal_cycles = total_macs / (DIM * DIM);

    printf("  cycles           : %llu\n", cycles);
    printf("  ideal cycles     : %llu\n", ideal_cycles);
    if (cycles == 0)
        printf("  utilization      : n/a (zero cycles - functional sim?)\n");
    else
        printf("  utilization      : %llu%%\n", 100 * ideal_cycles / cycles);
    printf("  RDMA_BYTES_REC   : %u\n", counter_read(0));
    printf("  WDMA_BYTES_SENT  : %u\n", counter_read(1));
    printf("  EXE_ACTIVE       : %u\n", counter_read(2));
    printf("  LOAD_DMA_WAIT    : %u\n", counter_read(3));
    printf("  STORE_DMA_WAIT   : %u\n", counter_read(4));
    printf("  SPAD_A_WAIT      : %u\n", counter_read(5));
    printf("  SPAD_B_WAIT      : %u\n", counter_read(6));
    printf("  RES_STATION_FULL : %u\n", counter_read(7));
}

// Word-at-a-time so the scalar setup doesn't dominate RTL simulation time.
static void fill_words(void *p, size_t bytes) {
    uint64_t *w = (uint64_t*)p;
    for (size_t i = 0; i < bytes / sizeof(uint64_t); i++)
        w[i] = (uint64_t)rand() * 0x0101010101010101ULL;
}

#if CHECK_RESULTS
static uint64_t checksum_C() {
    const uint64_t *w = (const uint64_t*)full_C;
    uint64_t sum = 0;
    for (size_t i = 0; i < sizeof(full_C) / sizeof(uint64_t); i++)
        sum += w[i];
    return sum;
}
#endif

static void run_matmul(size_t tile_I, size_t tile_J, size_t tile_K) {
    tiled_matmul(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B,
            NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            tile_I, tile_J, tile_K,
            A_TRANSPOSE, B_TRANSPOSE,
            false, false,
            0,
            WS);
}

static size_t isqrt(size_t n) {
    size_t r = 0;
    while ((r + 1) * (r + 1) <= n) r++;
    return r;
}

// Mirrors the seed + greedy growth loop in tiled_matmul_auto, which can't be
// used directly because PRINT_TILE makes it exit() before running the matmul.
static void print_auto_tiling(void) {
    const size_t max_spad_rows = BANK_NUM * BANK_ROWS / 2;
    const size_t max_acc_rows  = ACC_ROWS / 2;

    // Names must not collide with gemmini.h's db_* macros, which it does not undef.
    const size_t mats_in_part = ((BANK_NUM * BANK_ROWS / 2) / 2) / DIM;
    const size_t mats_in_acc  = (ACC_ROWS / 2) / DIM;
    const size_t seed_i_j     = isqrt(mats_in_acc);
    const size_t seed_k       = mats_in_part / seed_i_j;

    const size_t I_blocks = (MAT_DIM_I / DIM + (MAT_DIM_I % DIM != 0));
    const size_t J_blocks = (MAT_DIM_J / DIM + (MAT_DIM_J % DIM != 0));
    const size_t K_blocks = (MAT_DIM_K / DIM + (MAT_DIM_K % DIM != 0));

    size_t tile_I = I_blocks < seed_i_j ? I_blocks : seed_i_j;
    size_t tile_J = J_blocks < seed_i_j ? J_blocks : seed_i_j;
    size_t tile_K = K_blocks < seed_k   ? K_blocks : seed_k;

    printf("  seed             : tile_I=%u tile_J=%u tile_K=%u\n",
            (unsigned)tile_I, (unsigned)tile_J, (unsigned)tile_K);

    while (1) {
        bool increased = false;

        if (tiled_matmul_total_spad_rows(tile_I, tile_J+1, tile_K) <= max_spad_rows &&
            tiled_matmul_total_acc_rows(tile_I, tile_J+1) <= max_acc_rows &&
            tile_J + 1 <= J_blocks) { tile_J++; increased = true; }

        if (tiled_matmul_total_spad_rows(tile_I+1, tile_J, tile_K) <= max_spad_rows &&
            tiled_matmul_total_acc_rows(tile_I+1, tile_J) <= max_acc_rows &&
            tile_I + 1 <= I_blocks) { tile_I++; increased = true; }

        if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K+1) <= max_spad_rows &&
            tile_K + 1 <= K_blocks) { tile_K++; increased = true; }

        if (!increased) break;
    }

    printf("  library picks    : tile_I=%u tile_J=%u tile_K=%u\n",
            (unsigned)tile_I, (unsigned)tile_J, (unsigned)tile_K);
    printf("  spad rows used   : %u / %u\n",
            (unsigned)tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K),
            (unsigned)max_spad_rows);
    printf("  acc rows used    : %u / %u\n",
            (unsigned)tiled_matmul_total_acc_rows(tile_I, tile_J),
            (unsigned)max_acc_rows);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    printf("=== tile sweep ===\n");
    printf("I=%d J=%d K=%d  (blocks: %d x %d x %d, DIM=%d)\n",
            MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            MAT_DIM_I/DIM, MAT_DIM_J/DIM, MAT_DIM_K/DIM, DIM);
    printf("BANK_NUM=%d BANK_ROWS=%d ACC_ROWS=%d\n",
            BANK_NUM, BANK_ROWS, ACC_ROWS);
    printf("budgets (WS, double-buffered): spad_rows=%d acc_rows=%d\n",
            BANK_NUM*BANK_ROWS/2, ACC_ROWS/2);

    printf("filling matrices...\n");
    fill_words(full_A, sizeof(full_A));
    fill_words(full_B, sizeof(full_B));
    printf("done filling\n\n");

    setup_counters();

    // Untimed warm-up. Without it the first measured run pays cold cache, TLB
    // and DRAM row-buffer costs that later runs don't, which showed up as
    // auto and an identical explicit tiling differing by 3%.
    printf("warmup...\n");
    run_matmul(WARMUP_TILE_I, WARMUP_TILE_J, WARMUP_TILE_K);
    gemmini_fence();
    printf("warmup done\n\n");

    printf("--- tiled_matmul_auto (library's choice) ---\n");
    print_auto_tiling();
    counter_reset();
    uint64_t start = read_cycles();
    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B,
            NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            A_TRANSPOSE, B_TRANSPOSE,
            false, false,
            0,
            WS);
    gemmini_fence();
    uint64_t auto_cycles = read_cycles() - start;
    report(auto_cycles);

#if CHECK_RESULTS
    const uint64_t gold_sum = checksum_C();
    printf("  checksum         : %llu\n", gold_sum);
#endif
    printf("\n");

    for (size_t c = 0; c < NUM_CANDIDATES; c++) {
        const size_t tile_I = candidates[c][0];
        const size_t tile_J = candidates[c][1];
        const size_t tile_K = candidates[c][2];

        printf("--- tiled_matmul tile_I=%u tile_J=%u tile_K=%u ---\n",
                (unsigned)tile_I, (unsigned)tile_J, (unsigned)tile_K);
        printf("  spad rows used   : %u / %d\n",
                (unsigned)((tile_I*tile_K + tile_K*tile_J) * DIM),
                BANK_NUM*BANK_ROWS/2);
        printf("  acc rows used    : %u / %d\n",
                (unsigned)((tile_I*tile_J) * DIM), ACC_ROWS/2);

        counter_reset();
        start = read_cycles();
        run_matmul(tile_I, tile_J, tile_K);
        gemmini_fence();
        uint64_t cycles = read_cycles() - start;
        report(cycles);

        if (cycles < auto_cycles)
            printf("  vs auto          : %llu%% FASTER\n",
                    100 * (auto_cycles - cycles) / auto_cycles);
        else
            printf("  vs auto          : %llu%% slower\n",
                    100 * (cycles - auto_cycles) / auto_cycles);

#if CHECK_RESULTS
        const uint64_t sum = checksum_C();
        printf("  checksum         : %llu %s\n", sum,
                sum == gold_sum ? "(ok)" : "(MISMATCH!)");
#endif
        printf("\n");
    }

    printf("=== done ===\n");
    exit(0);
}
