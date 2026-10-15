# ChronoSentry

> Spatio-Temporal Memory Architecture for Autonomous Coding Agents. Built with PostgreSQL 16 & pgvector (`halfvec(1536)`).

ChronoSentry eliminates the **Temporal Amnesia / Stale Vector Problem** in modern AI coding agents. When an agent modifies or refactors a codebase, standard flat vector search frequently retrieves obsolete function signatures because semantic similarity alone ignores chronological recency and deprecation state.

ChronoSentry combines Partial HNSW index traversal with an automated AST supersession database trigger and a Two-Stage CTE recency re-ranking kernel directly inside PostgreSQL, cutting memory by 50% via `halfvec(1536)` and keeping evergreen foundational utilities fully preserved.

## Architecture

```text
               Incoming Coding Query / Agent Intent
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │       ChronoSentry Two-Stage CTE             │
         └──────────────────────┬───────────────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
   ┌───────────────────────────┐ ┌───────────────────────────┐
   │  Stage 1: Partial HNSW ANN│ │  Stage 2: Temporal Re-rank│
   │  - halfvec(1536) (50% RAM)│ │  - Mutability-aware decay │
   │  - O(log N) graph search  │ │  - Evergreen protection   │
   └─────────────┬─────────────┘ └─────────────┬─────────────┘
                 │                             │
                 └──────────────┬──────────────┘
                                │
                                ▼
                Top-K Zero-Stale Verified Context
```

## Quickstart

```bash
# 1. Clone repository
git clone https://github.com/Li-Qifeng/chronosentry.git
cd chronosentry

# 2. Start PostgreSQL 16 + pgvector container
docker compose up -d

# 3. Run empirical benchmark
pip install -r requirements.txt
python src/benchmark.py
```

## License
MIT
