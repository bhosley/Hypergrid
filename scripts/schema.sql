-- schema.sql
PRAGMA journal_mode=WAL;            -- better concurrency/crash resistance
PRAGMA synchronous=NORMAL;          -- good durability/throughput tradeoff

CREATE TABLE IF NOT EXISTS experiments (
    exp_id          INTEGER PRIMARY KEY,
    n_samples       INTEGER NOT NULL,
    num_agents      INTEGER NOT NULL,
    policy_type     TEXT NOT NULL,
    agent_sensors   TEXT,
    save_path       TEXT,
    -- config_hash     TEXT NOT NULL,      -- dedup key
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- CREATE UNIQUE INDEX IF NOT EXISTS idx_experiments_config_hash
--   ON experiments(config_hash);

CREATE TABLE IF NOT EXISTS train_runs (
    run_id          INTEGER PRIMARY KEY,    -- UUID
    exp_id          INTEGER NOT NULL,       -- Foreign Key
    sample_idx      INTEGER NOT NULL,       -- 0..n_samples-1
    status          TEXT NOT NULL CHECK(
        status IN ('pending','running','succeeded','failed','aborted')),
    started_at      TEXT,
    model_path      TEXT,                   -- checkpoint dir/file
    finished_at     TEXT
);

-- CREATE INDEX IF NOT EXISTS idx_train_status ON train_runs(status);

CREATE TABLE IF NOT EXISTS eval_runs (
    eval_id     INTEGER PRIMARY KEY, -- UUID
--   run_id           TEXT NOT NULL,    -- FK -> train_runs
    eval_type   TEXT NOT NULL,
    status      TEXT NOT NULL CHECK(
        status IN ('pending','running','succeeded','failed','aborted')),
    started_at  DATETIME,
    finished_at DATETIME
);

-- CREATE INDEX IF NOT EXISTS idx_eval_status ON eval_runs(status);
