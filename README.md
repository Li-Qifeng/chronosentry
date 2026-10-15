# ChronoSentry

> Spatio-Temporal Memory Architecture for Autonomous Coding Agents. Built with TimescaleDB & PostgreSQL pgvector (`halfvec(1536)`).

ChronoSentry eliminates the **Temporal Amnesia / Stale Vector Problem** in modern AI coding agents. When an agent modifies or refactors a codebase, standard flat vector search frequently retrieves obsolete function signatures because semantic similarity alone ignores chronological recency and deprecation state.

ChronoSentry combines HNSW index traversal with a monotonic exponential half-life decay kernel directly inside PostgreSQL, cutting memory by 50% via `halfvec(1536)` and reducing stale context hallucinations from 100% to 0%.

## Architecture

```text
               Incoming Coding Query / Agent Intent
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │      ChronoSentry Unified SQL Kernel         │
         │  Score = CosineSim * exp(-λ * Δt / τ) * Mask │
         └──────────────────────┬───────────────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
   ┌───────────────────────────┐ ┌───────────────────────────┐
   │    pgvector halfvec(1536) │ │   Temporal Partitioning   │
   │  - 50% RAM efficiency     │ │  - 1-day Hypertable Chunks│
   │  - HNSW (m=16, ef=64)     │ │  - Btree (session, time)  │
   └─────────────┬─────────────┘ └─────────────┬─────────────┘
                 │                             │
                 └──────────────┬──────────────┘
                                │
                                ▼
                Top-K Zero-Stale Verified Context
```

## Empirical Benchmarks

Based on 200 time-stamped context diffs across 14 simulated days of codebase refactoring:

| Metric | Vanilla Vector Search (Cosine Only) | ChronoSentry (Time-Decayed Halfvec) | Improvement |
| :--- | :---: | :---: | :---: |
| **Stale Code Hallucination Rate** | **100.0%** | **0.0%** | **100% Elimination** |
| **Median Latency** | 19.93 ms | 17.01 ms | Sub-20ms real-time |
| **Buffer Cache Hit Ratio** | - | 100% (240/240 shared hit) | Zero disk I/O |
| **Index Memory Footprint** | ~3.8 MB (FP32) | ~1.9 MB (FP16 halfvec) | **50% RAM reduction** |

## Quickstart

```bash
# 1. Clone repository
git clone https://github.com/Li-Qifeng/chronosentry.git
cd chronosentry

# 2. Start PostgreSQL + pgvector
docker compose up -d

# 3. Run empirical benchmark
pip install -r requirements.txt
python src/benchmark.py
```

## License
MIT
