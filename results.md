# Tiling Study: Auditing Gemmini's Auto-Tiler

## Question

Gemmini's software library picks a tile size automatically via `tiled_matmul_auto()`
when running a tiled matmul on data too large to fit on-chip in one pass. Does that
automatic choice actually run fastest, or does it just satisfy the capacity
constraints without regard to performance?

## Setup

- Matmul: 128×256×256 (8×16×16 in 16×16 blocks), weight-stationary (WS) dataflow.
- Hardware budgets: scratchpad 8192 rows, accumulator 512 rows.
- `tiled_matmul_auto` picks `tile_I=5, tile_J=6, tile_K=16` — 93% accumulator
  utilization, 34% scratchpad utilization.
- Compared against six candidates: `(5,6,16)` [reproduces auto's own pick],
  `(4,8,16)`, `(8,4,16)`, `(4,4,16)`, `(2,16,16)`, `(6,5,16)`.
- Measured on Verilator RTL simulation (`GemminiRocketConfig`, `DIM=16`).
  Spike was used only for fast functional/correctness checks — it does not
  model timing, so its cycle counts and hardware counters are meaningless.
- Benchmark: `bareMetalC/tile_sweep.c`.

## Reproducing this

Build the benchmark:

```bash
cd generators/gemmini/software/gemmini-rocc-tests
./build.sh bareMetalC
```

Run it directly against the Verilator simulator (not through `make run-binary`,
which pipes output through `tee` and defeats line-buffering — see note below):

```bash
cd ../../../../sims/verilator   # sims/verilator from repo root

BIN=../../generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/tile_sweep-baremetal

./simulator-chipyard.harness-GemminiRocketConfig \
  +permissive \
  +dramsim +dramsim_ini_dir=../../generators/testchipip/src/main/resources/dramsim2_ini \
  +max-cycles=200000000 \
  +loadmem=$BIN \
  +permissive-off \
  $BIN
```

Notes:
- `+loadmem=$BIN` loads the ELF directly into simulated DRAM. Without it, boot
  takes 6+ minutes instead of ~20 seconds (the binary otherwise streams in
  through the serial link word-by-word).
- Running the simulator directly, rather than through `make ... | tee ...`,
  matters because output is block-buffered (4KB) rather than line-buffered
  when stdout is a pipe — with `tee` in the path you can see no output for
  a long time even though the program is running correctly.



## Measurement validity

Running the *identical* tiling twice — once via `tiled_matmul_auto` and once
by calling `tiled_matmul` explicitly with the same `(5,6,16)` — initially gave
cycle counts differing by ~3% (38,460 vs. 37,330), because the first run in
the program pays cold-cache, cold-TLB, and cold-DRAM-row-buffer costs that
later runs don't.

Adding an untimed warm-up matmul before any timed run brought this down to
~0.34% (37,520 vs. 37,392). All results below use the warm-up. Differences
between tilings smaller than ~1% are treated as noise; differences above that
are treated as real.

## Results

| Tiling (I,J,K)      | Cycles | vs. auto | Utilization | RDMA bytes | EXE_ACTIVE | LOAD_DMA_WAIT |
|---------------------|--------|----------|-------------|------------|------------|---------------|
| auto → (5,6,16)      | 37,520 | —        | 87%         | 557,056    | 33,159     | 93            |
| (5,6,16) explicit    | 37,392 | +0.3%    | 87%         | 557,056    | 33,274     | 96            |
| (4,8,16)             | 38,275 | -2.0%    | 85%         | 425,984    | 33,558     | 0             |
| (8,4,16)             | 38,278 | -2.0%    | 85%         | 425,984    | 33,543     | 0             |
| **(4,4,16)**         | **36,265** | **+3.3%** | **90%** | 688,128    | 32,581     | 0             |
| (2,16,16)            | 39,876 | -6.3%    | 82%         | 425,984    | 35,157     | 0             |
| (6,5,16)             | 38,068 | -1.5%    | 86%         | 532,480    | 34,256     | 230           |

`vs. auto` is relative to auto's 37,520 cycles; positive = faster. All seven
runs produced an identical output checksum, confirming every tiling computes
the same numerical result regardless of schedule.

Full raw output from the run behind the results below: [`tile_sweep_run.log`](./tile_sweep_run.log).

## Finding 1: auto is not optimal

`(4,4,16)` completes in 36,265 cycles versus auto's 37,520 — a **3.3%
improvement**, well above the ~1% noise floor. Auto's heuristic (a square
`sqrt`-based seed, greedily grown to fill available capacity) does not find
this tiling, because reaching `(4,4,16)` from the `(5,5,16)` seed requires
*shrinking* `tile_I` and `tile_J` first, and the greedy loop only ever grows.

## Finding 2: the obvious explanation is backwards

The natural hypothesis is that a tiling wins by moving less data. The data
says the opposite. Sorting by RDMA bytes fetched:
```
425,984 bytes  →  38,275 / 38,278 cycles   (slowest)
532,480 bytes  →  38,068 cycles
557,056 bytes  →  37,392 / 37,520 cycles
688,128 bytes  →  36,265 cycles            (fastest)
```

Memory traffic and runtime are *inversely* correlated. `(4,8,16)`/`(8,4,16)`
fetch 23% less data than auto yet run ~2% slower. `(4,4,16)` fetches the
*most* data of any tiling tested — 61% more than `(4,8,16)` — yet is the
fastest.

## Why

`LOAD_DMA_WAIT_CYCLE` stays under 230 cycles out of ~37,000 for every single
tiling, and execution-unit utilization sits at 82-90% throughout. The
accelerator is essentially never waiting on memory — this workload is
**compute-bound**, not memory-bound, so optimizing for less memory traffic
optimizes a resource that was never the bottleneck.

What actually predicts runtime is `EXE_ACTIVE_CYCLE` and
`RESERVATION_STATION_FULL_CYCLES`, both lowest for the winning `(4,4,16)`
tiling. Smaller tiles appear to keep the execution pipeline fed more
efficiently — plausibly less accumulator fill/drain overhead per unit of
work — and that effect dominates over any traffic savings from larger tiles.
The counters establish the correlation; pinning down the exact microarchitectural
mechanism would need RTL-level waveform inspection, which is out of scope here.

## Scope and limits

- Only 6 of the 69 tilings that satisfy the capacity constraints at this
  matrix size were tested. `(4,4,16)` is the best tiling *found*, not
  necessarily the global optimum — a full sweep of the legal tiling space is
  the natural next step.
- This result is specific to a 128×256×256 matmul on the default Gemmini
  config. Whether the same compute-bound regime (and therefore the same
  "less traffic ≠ faster" result) holds for other matrix shapes or
  configurations is untested.
