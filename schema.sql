-- ChronoSentry: Spatio-Temporal Memory Architecture for Autonomous Coding Agents
-- Built for PostgreSQL 16 + TimescaleDB + pgvector halfvec(1536)

CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing structures if re-initializing
DROP FUNCTION IF EXISTS match_code_context_temporal(HALFVEC, TEXT, FLOAT, FLOAT, INT);
DROP TABLE IF EXISTS code_context_events CASCADE;
DROP TABLE IF EXISTS agent_sessions CASCADE;

CREATE TABLE agent_sessions (
    session_id TEXT PRIMARY KEY,
    repository_name TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Core Temporal-Vector Log
CREATE TABLE code_context_events (
    event_id BIGSERIAL,
    recorded_at TIMESTAMPTZ NOT NULL,
    session_id TEXT NOT NULL REFERENCES agent_sessions(session_id),
    file_path TEXT NOT NULL,
    symbol_name TEXT,
    event_type TEXT NOT NULL, -- 'declaration', 'refactor', 'deprecation', 'execution_trace'
    code_diff TEXT NOT NULL,
    superseded BOOLEAN NOT NULL DEFAULT FALSE,
    embedding HALFVEC(1536) NOT NULL
);

-- Attempt TimescaleDB hypertable conversion if extension is present
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb') THEN
        BEGIN
            CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
            PERFORM create_hypertable('code_context_events', 'recorded_at', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
            RAISE NOTICE 'TimescaleDB Hypertable initialized for code_context_events';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'TimescaleDB extension not loaded in container, using native PostgreSQL table.';
        END;
    END IF;
END $$;

-- Create HNSW Cosine Index over halfvec(1536) for 50% RAM reduction
CREATE INDEX idx_code_context_halfvec_hnsw 
ON code_context_events 
USING hnsw (embedding halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Composite btree index for temporal filtering and session lookups
CREATE INDEX idx_code_context_session_time 
ON code_context_events (session_id, recorded_at DESC);

CREATE INDEX idx_code_context_file_path 
ON code_context_events (file_path, recorded_at DESC);

-- Custom Function: Time-Decayed Hybrid Similarity Search
-- Formula: Score = (1 - (embedding <=> query_vec)) * exp(-0.693 * (dt_hours / half_life_hours)) * (1 - superseded::int)
CREATE OR REPLACE FUNCTION match_code_context_temporal(
    query_vec HALFVEC(1536),
    match_session TEXT,
    half_life_hours FLOAT DEFAULT 72.0,
    match_threshold FLOAT DEFAULT 0.10,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    event_id BIGINT,
    recorded_at TIMESTAMPTZ,
    file_path TEXT,
    symbol_name TEXT,
    event_type TEXT,
    code_diff TEXT,
    cosine_sim FLOAT,
    temporal_weight FLOAT,
    final_score FLOAT
)
LANGUAGE sql STABLE AS $$
    SELECT 
        c.event_id,
        c.recorded_at,
        c.file_path,
        c.symbol_name,
        c.event_type,
        c.code_diff,
        (1 - (c.embedding <=> query_vec))::FLOAT AS cosine_sim,
        EXP(-0.693 * (EXTRACT(EPOCH FROM (NOW() - c.recorded_at)) / 3600.0) / half_life_hours)::FLOAT AS temporal_weight,
        ((1 - (c.embedding <=> query_vec)) * EXP(-0.693 * (EXTRACT(EPOCH FROM (NOW() - c.recorded_at)) / 3600.0) / half_life_hours))::FLOAT AS final_score
    FROM code_context_events c
    WHERE c.session_id = match_session
      AND c.superseded = FALSE
      AND (1 - (c.embedding <=> query_vec)) > match_threshold
    ORDER BY final_score DESC
    LIMIT match_count;
$$;
