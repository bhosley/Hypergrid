import sqlite3
from pathlib import Path
import argparse
from itertools import product
from datetime import datetime
import json

# from import main as train


def train(num_agents, env_config, make_homo, **kwargs):
    print(
        "Running training as"
        "main(make_homo={},num_agents={},env_config={})".format(
            make_homo, num_agents, env_config
        )
    )
    # model_path = results[0].checkpoint.path
    results = ["dummy_path"]
    return results


SCHEMA_PATH = Path("schema.sql")
DB_PATH = Path("test.db")


def _C2_factors():
    """Fill Experiment Factors."""
    n_agent_factors = [4]
    policy_factors = ["default_het", "induced_hom"]
    agent_sensors = {
        "complete": None,
        "intersecting_span": {
            0: [1, 1, 0, 0, 1, 1],
            1: [0, 1, 1, 0, 1, 1],
            2: [0, 0, 1, 1, 1, 1],
            3: [1, 0, 0, 1, 1, 1],
        },
        "disjoint_span": {
            0: [1, 0, 0, 0, 1, 1],
            1: [0, 1, 0, 0, 1, 1],
            2: [0, 0, 1, 0, 1, 1],
            3: [0, 0, 0, 1, 1, 1],
        },
        "incomplete": {
            0: [1, 0, 0, 0, 1, 1],
            1: [0, 1, 0, 0, 1, 1],
            2: [0, 0, 1, 0, 1, 1],
            3: [0, 1, 0, 0, 1, 1],
        },
    }
    factors = list(product(n_agent_factors, policy_factors, agent_sensors))
    return [
        (a, p, json.dumps({"config": s, "agent_sensors": agent_sensors[s]}))
        for a, p, s in factors
    ]


def populate_experiments(
    db_path: Path | str,
    exp_path: Path | str,
    samples: int,
):
    """Populate the Experiments Table"""
    # Connect to DB
    conn = connect_to_DB(db_path)
    cursor = conn.cursor()

    try:
        # Batch query insert experiment factors
        conn.execute("BEGIN")
        for factor_list in _C2_factors():
            # (n_agent,policy_set,agent_sensors) = factor_list
            # Validate Uniqueness:
            cursor.execute(
                """
                SELECT * FROM experiments
                WHERE num_agents=? AND policy_type=? AND agent_sensors=?
                """,
                factor_list,
            )
            num = len(cursor.fetchall())
            if num < 1:
                cursor.execute(
                    """INSERT INTO
                    experiments(n_samples, num_agents, policy_type, agent_sensors, exp_path)
                    VALUES (?,?,?,?,?)""",
                    (samples, *factor_list, exp_path),
                )
        # Commit the transaction
        conn.execute("COMMIT")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        # Roll back the transaction if an error occurs
        conn.execute("ROLLBACK")
    finally:
        if conn:
            conn.close()


def populate_train_samples(
    db_path: Path | str,
):
    conn = connect_to_DB(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM experiments")

        exp_rows = [(rw["exp_id"], rw["n_samples"]) for rw in cursor.fetchall()]
        for exp_id, exp_samples in exp_rows:
            # Count Planned Samples
            cursor.execute(
                "SELECT sample_idx FROM train_runs WHERE exp_id = {}".format(
                    exp_id
                )
            )
            # Difference between desired and planned samples
            existing_ids = [_["sample_idx"] for _ in cursor.fetchall()]
            ids_to_make = set(range(1, exp_samples + 1)).difference(
                existing_ids
            )
            # Populate missing sample indexes to reach desired sample size
            for sample_id in ids_to_make:
                cursor.execute(
                    """
                    INSERT INTO
                    train_runs(exp_id, sample_idx, status)
                    VALUES (?,?,?)
                    """,
                    (exp_id, sample_id, "pending"),
                )
                # Commit the transaction
                conn.execute("COMMIT")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        # Roll back the transaction if an error occurs
        conn.execute("ROLLBACK")
    finally:
        if conn:
            conn.close()


def train_sample(
    db_path: Path | str,
):
    """"""
    conn = connect_to_DB(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        # Randomly select a scheduled sample
        run = cursor.execute(
            """
            SELECT * FROM train_runs
            WHERE status='pending'
            ORDER BY RANDOM()
            """
        ).fetchone()

        if not run:
            # We have no more runs
            return

        run_id = run["run_id"]
        exp_id = run["exp_id"]
        # Set to running
        cursor.execute(
            """
            UPDATE train_runs
            SET status='running', started_at=?
            WHERE run_id=?""",
            (datetime.now().isoformat(), run_id),
        )
        # Commit the transaction
        conn.execute("COMMIT")
        # Get exp config
        conf = cursor.execute(
            "SELECT * FROM experiments WHERE exp_id={}".format(exp_id)
        ).fetchone()
        if conn:
            conn.close()

        # Clean config
        num_agents = int(conf["num_agents"])
        make_homo = conf["policy_type"] == "induced_hom"
        env_config = {
            "agent_sensors": {
                int(k): v for k, v in json.loads(conf["agent_sensors"]).items()
            }
        }
        # Call Train, collect path for recording
        results = train(
            num_agents=num_agents,
            env_config=env_config,
            make_homo=make_homo,
        )
        model_path = results[0].checkpoint.path

        # When finished training
        conn = connect_to_DB(db_path)
        # conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE train_runs
            SET status='succeeded', model_path=?, finished_at=?
            WHERE run_id=?""",
            (model_path, datetime.now().isoformat(), run_id),
        )
        # Commit the transaction
        conn.execute("COMMIT")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        # Roll back the transaction if an error occurs
        conn.execute("ROLLBACK")
    finally:
        if conn:
            conn.close()


def clean():
    NotImplementedError
    # TODO: When cleaning stale datetime.fromisoformat(_)


def connect_to_DB(
    db_path: Path | str,
):
    """"""
    # Connect or create
    db_path = Path(db_path)
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Instantiate connection
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")

    return conn


def make_new_DB(
    db_path: Path | str,
    schema: str = None,
    schema_path: str | Path = None,
):
    """Builds a DB from supplied schema or path"""
    if not schema and not schema_path:
        raise ValueError("This function requires a schema or schema path.")
    if schema and schema_path:
        raise ValueError("This function takes one of schema or schema path.")
    # Connect to DB
    conn = connect_to_DB(db_path)
    cursor = conn.cursor()
    # Build db from input directions
    if schema:
        cursor.execute(schema)
    if schema_path:
        with open(Path(schema_path), "r") as f:
            conn.executescript(f.read())
    # Confirm that the table has been created
    print("Table is Ready")
    # Close the connection to the database
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-M",
        "--make",
        type=str,
        nargs="?",
        const=True,
        default=False,
        help="Make a new DB, optionally provide path to a schema.",
    )
    parser.add_argument(
        "-P",
        "--populate",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="Populate.",
    )
    parser.add_argument(
        "-R",
        "--run",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="Run.",
    )
    args = parser.parse_args()

    if args.make:
        make_new_DB(
            db_path=DB_PATH,
            schema_path=SCHEMA_PATH,
        )

    if args.populate:
        populate_experiments(db_path=DB_PATH, exp_path="", samples=3)
        populate_train_samples(db_path=DB_PATH)

    for _ in range(args.run):
        train_sample(db_path=DB_PATH)
