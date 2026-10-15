-- ChronoSentry: Spatio-Temporal Memory Architecture for Autonomous Coding Agents
-- Production Engine for PostgreSQL 16 + pgvector halfvec(1536)

CREATE EXTENSION IF NOT EXISTS vector;

-- Cleanup existing objects
DROP TRIGGER IF EXISTS trg_auto_supersede ON code_context_events;
DROP FUNCTION IF EXISTS auto_supersede_symbol CASCADE;
DROP FUNCTION IF EXISTS match_code_context_two_stage CASCADE;
DROP FUNCTION IF EXISTS match_code_context_temporal CASCADE;
DROP TABLE IF EXISTS code_context_events CASCADE;
DROP TABLE IF EXISTS agent_sessions CASCADE;

-- Agent Sessions & Repositories
CREATE TABLE agent_sessions (
    session_id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Core Temporal-Vector Log
CREATE TABLE code_context_events (
    event_id BIGSERIAL PRIMARY KEY,
    recorded_at TIMESTAMPTZ NOT NULL,
    repository_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    symbol_name TEXT,
    event_type TEXT NOT NULL, -- 'declaration', 'refactor', 'deprecation', 'execution_trace'
    mutability_class TEXT NOT NULL DEFAULT 'standard' CHECK (mutability_class IN ('immutable_core', 'standard', 'transient_diff')),
    code_diff TEXT NOT NULL,
    superseded BOOLEAN NOT NULL DEFAULT FALSE,
    embedding HALFVEC(1536) NOT NULL
);

-- Automated AST Supersession Trigger
-- When a new implementation of a symbol in the same file is committed, older entries are marked superseded
CREATE OR REPLACE FUNCTION auto_supersede_symbol()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.symbol_name IS NOT NULL AND NEW.superseded = FALSE THEN
        UPDATE code_context_events
        SET superseded = TRUE
        WHERE repository_id = NEW.repository_id
          AND file_path = NEW.file_path
          AND symbol_name = NEW.symbol_name
          AND event_id <> NEW.event_id
          AND superseded = FALSE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_auto_supersede
BEFORE INSERT ON code_context_events
FOR EACH ROW EXECUTE FUNCTION auto_supersede_symbol();

-- HNSW Cosine Index over halfvec(1536) for 50% RAM efficiency
CREATE INDEX idx_code_context_halfvec_hnsw 
ON code_context_events 
USING hnsw (embedding halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Composite btree index for repo + temporal queries
CREATE INDEX idx_code_context_repo_time 
ON code_context_events (repository_id, recorded_at DESC);

CREATE INDEX idx_code_context_file_symbol 
ON code_context_events (repository_id, file_path, symbol_name);

-- Two-Stage Retrieval CTE: Rescues HNSW Index Traversal
-- Stage 1: Fast HNSW Approximate Nearest Neighbor Graph Traversal (O(log N))
-- Stage 2: Adaptive In-Memory Spatio-Temporal Re-ranking with Evergreen Mutability Classes
CREATE OR REPLACE FUNCTION match_code_context_two_stage(
    query_vec HALFVEC(1536),
    target_repo TEXT,
    half_life_hours FLOAT DEFAULT 72.0,
    candidate_k INT DEFAULT 40,
    final_limit INT DEFAULT 5
)
RETURNS TABLE (
    event_id BIGINT,
    recorded_at TIMESTAMPTZ,
    file_path TEXT,
    symbol_name TEXT,
    mutability_class TEXT,
    code_diff TEXT,
    cosine_sim FLOAT,
    temporal_weight FLOAT,
    final_score FLOAT
)
LANGUAGE sql STABLE AS $$
    WITH candidate_vectors AS (
        -- Stage 1: Native HNSW Vector Graph Traversal
        SELECT 
            c.event_id,
            c.recorded_at,
            c.file_path,
            c.symbol_name,
            c.mutability_class,
            c.code_diff,
            (1 - (c.embedding <=> query_vec))::FLOAT AS cosine_sim
        FROM code_context_events c
        WHERE c.repository_id = target_repo
          AND c.superseded = FALSE
        ORDER BY c.embedding <=> query_vec
        LIMIT candidate_k
    )
    -- Stage 2: In-Memory Spatio-Temporal Re-ranking
    SELECT 
        v.event_id,
        v.recorded_at,
        v.file_path,
        v.symbol_name,
        v.mutability_class,
        v.code_diff,
        v.cosine_sim,
        (CASE 
            WHEN v.mutability_class = 'immutable_core' THEN 1.0::FLOAT
            WHEN v.mutability_class = 'transient_diff' THEN EXP(-0.693 * (EXTRACT(EPOCH FROM (NOW() - v.recorded_at)) / 3600.0) / (half_life_hours * 0.25))::FLOAT
            ELSE EXP(-0.693 * (EXTRACT(EPOCH FROM (NOW() - v.recorded_at)) / 3600.0) / half_life_hours)::FLOAT
        END) AS temporal_weight,
        (v.cosine_sim * (CASE 
            WHEN v.mutability_class = 'immutable_core' THEN 1.0::FLOAT
            WHEN v.mutability_class = 'transient_diff' THEN EXP(-0.693 * (EXTRACT(EPOCH FROM (NOW() - v.recorded_at)) / 3600.0) / (half_life_hours * 0.25))::FLOAT
            ELSE EXP(-0.693 * (EXTRACT(EPOCH FROM (NOW() - v.recorded_at)) / 3600.0) / half_life_hours)::FLOAT
        END))::FLOAT AS final_score
    FROM candidate_vectors v
    ORDER BY final_score DESC
    LIMIT final_limit;
$$;
