from __future__ import annotations

import sqlite3
import json
from datetime import datetime
from pathlib import Path

from scripts.db_manager import connect_to_DB
from scripts.eval import supplemental as supp_eval

query_create_shufnov = """
    CREATE TABLE IF NOT EXISTS eval_shufnov (
        eval_id     INTEGER PRIMARY KEY,    -- UUID
        exp_id      INTEGER NOT NULL,       -- FK
        run_id      INTEGER NOT NULL,       -- FK
        status      TEXT NOT NULL CHECK(
            status IN ('pending','running','succeeded','failed','aborted')),
        started_at  TEXT,
        finished_at TEXT
    );
    """
query_unscheduled_shufnov = """
    SELECT tr.run_id, tr.exp_id
    FROM train_runs AS tr
    WHERE tr.status = 'succeeded'
    AND NOT EXISTS (
        SELECT 1 FROM eval_shufnov er
        WHERE er.run_id = tr.run_id
    )
    """
query_add_shufnov = """
    INSERT INTO eval_shufnov (exp_id, run_id, status)
    VALUES (?,?,?)
    """
query_get_rand_shufnov = """
    SELECT er.eval_id, tr.model_path
    FROM eval_shufnov AS er
    INNER JOIN train_runs AS tr
    ON er.run_id = tr.run_id
    WHERE er.status = 'pending'
    ORDER BY RANDOM()
    """
query_update_shufnov = """
    UPDATE eval_shufnov
    SET status=?, started_at=?
    WHERE eval_id=?
    """
query_get_shufnov_conf = "SELECT * FROM eval_shufnov WHERE eval_id=?"
query_get_exp_config = "SELECT * FROM experiments WHERE exp_id=?"
query_get_run_config = "SELECT * FROM train_runs WHERE run_id=?"


def eval_shuffle_and_novel(
    db_path: Path,
    eval_reps: int = 1,
    eval_sample_size: int = 30,
    configs: dict = {},
):
    """ """
    # Get Connection
    conn = connect_to_DB(db_path)
    cursor = conn.cursor()
    print("Verifying Supplemental Evaluations Table...")

    # Make table if it doesn't exist
    try:
        cursor.execute(query_create_shufnov)
    except sqlite3.Error as e:
        # Roll back the transaction if an error occurs
        print(f"An error occurred while making table: {e}")
        conn.execute("ROLLBACK")

    # Populate table if it doesn't exist
    try:
        entries = 0
        cursor.execute(query_unscheduled_shufnov)
        unscheduled_evals = cursor.fetchall()
        for eval_row in unscheduled_evals:
            values = (eval_row["exp_id"], eval_row["run_id"], "pending")
            cursor.execute(query_add_shufnov, values)
            entries += 1
        print(f"Adding {entries} new evaluations")
        if entries > 0:
            conn.execute("COMMIT")
    except sqlite3.Error as e:
        # Roll back the transaction if an error occurs
        print(f"An error occurred while populating table: {e}")
        conn.execute("ROLLBACK")

    # Begin running
    print("Beginning Supplemental Evaluations..")
    try:
        for _ in range(eval_reps):
            # Randomly select a scheduled sample
            eval_entry = cursor.execute(query_get_rand_shufnov).fetchone()
            if not eval_entry:
                print("Did not find any more evaluations to run. Exiting...")
                if conn:
                    conn.close()
                return
            eval_id = eval_entry["eval_id"]
            # Set to running - commit immediately
            start_time = datetime.now().isoformat()
            cursor.execute(
                query_update_shufnov, ("running", start_time, eval_id)
            )
            conn.execute("COMMIT")
            # Get eval info needed
            eval_conf = {}
            row = cursor.execute(query_get_shufnov_conf, (eval_id,)).fetchone()
            exp_id = row["exp_id"]
            run_id = row["run_id"]
            # Get experiment info
            exp_conf = cursor.execute(
                query_get_exp_config, (exp_id,)
            ).fetchone()
            eval_conf["num_agents"] = exp_conf["num_agents"]
            eval_conf["policy_type"] = exp_conf["policy_type"]
            sensors = json.loads(exp_conf["agent_sensors"])
            eval_conf["sensor_config"] = sensors["config"]
            eval_conf["agent_sensors"] = (
                {int(k): v for k, v in sensors["agent_sensors"].items()}
                if sensors["agent_sensors"]
                else None
            )
            # Get directory
            run_conf = cursor.execute(
                query_get_run_config, (run_id,)
            ).fetchone()
            eval_conf["load_dir"] = run_conf["model_path"]
            eval_conf["episodes"] = eval_sample_size
            # Release the database for concurrent runners
            if conn:
                conn.close()

            supp_eval(**eval_conf, **configs)

            end_time = datetime.now().isoformat()
            # When finished training reopen DB connection
            conn = connect_to_DB(db_path)
            cursor = conn.cursor()
            cursor.execute(
                query_update_shufnov, ("succeeded", end_time, eval_id)
            )
            # Commit the transaction
            conn.execute("COMMIT")

    except sqlite3.Error as e:
        # Roll back the transaction if an error occurs
        print(f"An error occurred while populating table: {e}")
        conn.execute("ROLLBACK")

    if conn:
        conn.close()
