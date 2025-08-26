import argparse
import json
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from itertools import product
from pathlib import Path

from scripts.train import main as train


def main(args, **kwargs):
    """ """
    # Version Specific:
    vers = "v1"
    train_configs = {
        "wandb": True,
        "num_timesteps": 1e8,
        "num_workers": 8,
        "project_name": f"hypergrid_{vers}",
    }
    # General use:
    # TODO: recover the better schema discoverer.
    load_dotenv()
    schema_path = args.schema_path if args.schema_path else Path("./schema.sql")
    share_path = Path(
        os.path.relpath(os.getenv("SHARE_PATH"), Path().cwd().resolve())
    ).joinpath(vers)
    db_path = share_path.joinpath("exp_manager.db")
    model_path = share_path.joinpath("ray_results")

    if args.make:
        make_new_DB(db_path=db_path, schema_path=schema_path)
    if args.clean:
        clean(db_path=db_path, minutes=args.clean, test=args.test)
    if args.populate:
        populate_experiments(
            db_path=db_path,
            save_dir=model_path,
            samples=args.populate,
            factors=_C2_factors(),
        )
        populate_train_samples(db_path=db_path)
    for _ in range(args.run):
        train_replication(db_path=db_path, train_configs=train_configs)
    # for _ in range(args.eval):


# --- --- #


def connect_to_DB(
    db_path: Path | str,
) -> sqlite3.Connection:
    """
    Connects to the database at the supplied path.

    Parameters
    ----------
    db_path : Path or string
        Path-like address to the database to be connected to.

    Returns
    -------
    connection : sqlite3.Connection
        SQLite object connection to the database.
    """
    # Connect or create
    db_path = Path(db_path)
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Instantiate connection
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.row_factory = sqlite3.Row

    return conn


# --- --- #


def make_new_DB(
    db_path: Path | str,
    schema: str = None,
    schema_path: Path | str = None,
) -> Path:
    """
    Builds a DB from supplied schema or path.

    Parameters
    ----------
    db_bath : Path or string
        The full path var directing to the manager database.
    schema : string
        A plain string SQL query based schema
    schema_path : Path or string
        Path-like location of a .sql file containing a schema for the database.
    """
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
    # Announce that the table has been created
    print("Table is Ready")
    # Close the connection to the database
    if conn:
        conn.close()
    return db_path


def clean(
    db_path: Path,
    minutes: int = 60,
    test: bool = False,
) -> None:
    """
    Remove stale replication entries from the database.

    Parameters
    ----------
    db_path : Path or string
        The full path var directing to the manager database.
    minutes : int
        Minimum age for an entry to be considered stale.
    test: bool
        Finds, counts, and reports stale entries without removing them.
    """
    # Get Connection
    conn = connect_to_DB(db_path)
    cursor = conn.cursor()
    # Pre-defined queries.
    query_stale = """
        SELECT run_id, started_at
        FROM train_runs
        WHERE status = 'running'
        AND julianday('now') - julianday(started_at, 'utc') > ?/1440.0
        """
    query_reset_rec = """
        UPDATE train_runs
        SET status='pending', started_at=NULL
        WHERE run_id=?
        """
    # Collect stale records
    stale_recs = cursor.execute(query_stale, (minutes,)).fetchall()
    count = 0
    for rec in stale_recs:
        # Try to reset the stale records
        if not test:
            try:
                cursor.execute(query_reset_rec, (rec["run_id"],))
                conn.execute("COMMIT")
                count += 1
            except sqlite3.Error as e:
                # Roll back the transaction if an error occurs
                print(f"An error occurred: {e}")
                conn.execute("ROLLBACK")
    if conn:
        conn.close()
    print(
        "Found {} stale records, successfully removed {}.".format(
            len(stale_recs), count
        )
    )


def populate_experiments(
    db_path: Path | str,
    factors: list,
    save_dir: Path | str = "",
    samples: int = 1,
):
    """Populate the Experiments Table"""
    # Connect to DB
    conn = connect_to_DB(db_path)
    cursor = conn.cursor()
    # Pre-define queries.
    query_existing = """
        SELECT * FROM experiments
        WHERE num_agents=? AND policy_type=? AND agent_sensors=?
        """
    query_add_exp = """
        INSERT INTO
        experiments(n_samples, num_agents, policy_type, agent_sensors, save_path)
        VALUES (?,?,?,?,?)
        """
    try:
        # Batch query insert experiment factors
        conn.execute("BEGIN")
        for factor_list in factors:
            # Validate Uniqueness before adding:
            cursor.execute(query_existing, factor_list)
            num = len(cursor.fetchall())
            if num < 1:
                cursor.execute(query_add_exp, (samples, *factor_list, save_dir))
        # Commit the transaction
        conn.execute("COMMIT")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        # Roll back the transaction if an error occurs
        conn.execute("ROLLBACK")
    finally:
        if conn:
            conn.close()
    # TODO: Populate experiment could be a bit more flexible (dictionary style)


def populate_train_samples(
    db_path: Path | str,
):
    """
    Pre-populate replications for each sample
    based on the data in the experiments table.

    Parameters
    ----------
    db_path : Path or string
        The full path var directing to the manager database.
    """
    # Connect
    conn = connect_to_DB(db_path)
    cursor = conn.cursor()
    # Pre-defined queries.
    query_samples = "SELECT sample_idx FROM train_runs WHERE exp_id = ?"
    query_add_sample = """
        INSERT INTO
        train_runs(exp_id, sample_idx, status)
        VALUES (?,?,?)
        """
    try:
        # Get experiment information
        cursor.execute("SELECT * FROM experiments")
        exp_rows = [(r["exp_id"], r["n_samples"]) for r in cursor.fetchall()]
        for exp_id, n_samples in exp_rows:
            # Collect indexes of samples by experiment ID.
            cursor.execute(query_samples, (exp_id,))
            # Difference between desired and planned samples
            existing_ids = [_["sample_idx"] for _ in cursor.fetchall()]
            ids_to_make = set(range(1, n_samples + 1)).difference(existing_ids)
            # Populate missing sample indexes to reach desired sample size
            for sample_id in ids_to_make:
                cursor.execute(query_add_sample, (exp_id, sample_id, "pending"))
                # Commit the immediately
                conn.execute("COMMIT")
    except sqlite3.Error as e:
        # Roll back the transaction if an error occurs
        print(f"An error occurred: {e}")
        conn.execute("ROLLBACK")
    finally:
        if conn:
            conn.close()


def train_replication(db_path: Path | str, train_configs: dict = {}):
    """
    Select and reserve a replication entry and perform training with
    the associated parameters. Pick randomly from replications that have
    not yet be run.

    Parameters
    ----------
    db_path : Path or string
        The full path var directing to the manager database.
    """
    # TODO: Add specific exp or rep support.
    # Get Connection
    conn = connect_to_DB(db_path)
    cursor = conn.cursor()
    query_get_random = """
        SELECT * FROM train_runs
        WHERE status='pending'
        ORDER BY RANDOM()
        """
    query_set_running = """
        UPDATE train_runs
        SET status='running', started_at=?
        WHERE run_id=?
        """
    query_get_exp_config = "SELECT * FROM experiments WHERE exp_id=?"
    query_set_complete = """
        UPDATE train_runs
        SET status='succeeded', model_path=?, finished_at=?
        WHERE run_id=?
        """
    try:
        # Randomly select a scheduled sample
        run = cursor.execute(query_get_random).fetchone()
        if not run:
            # We have no more runs
            return
        run_id = run["run_id"]
        exp_id = run["exp_id"]
        # Set to running - commit immediately
        cursor.execute(query_set_running, (datetime.now().isoformat(), run_id))
        conn.execute("COMMIT")
        # Get exp config
        row = cursor.execute(query_get_exp_config, (exp_id,)).fetchone()
        # Release the database for concurrent runners
        if conn:
            conn.close()

        config = _rep_row_to_config(row)
        results = train(**config, **train_configs)

        # Clean config
        # num_agents = int(conf["num_agents"])
        # make_homo = conf["policy_type"] == "induced_hom"
        # sensors = json.loads(conf["agent_sensors"])
        # env_config = {
        #     "agent_sensors": {
        #         int(k): v for k, v in sensors["agent_sensors"].items()
        #     }
        #     if sensors["agent_sensors"]
        #     else None
        # }
        # Call Train, collect path for recording
        # results = train(
        #     num_agents=num_agents,
        #     env_config=env_config,
        #     make_homo=make_homo,
        #     sensor_conf=sensors["config"],
        #     # TODO: add a switch for this
        #     wandb=True,
        #     num_timesteps=1e8,
        #     num_workers=8,
        # )

        # When finished training reopen DB connection
        conn = connect_to_DB(db_path)
        cursor = conn.cursor()
        # Update with path to complete policies and time of completion
        model_path = results[0].checkpoint.path
        end_time = datetime.now().isoformat()
        cursor.execute(query_set_complete, (model_path, end_time, run_id))
        # Commit the transaction
        conn.execute("COMMIT")
    except sqlite3.Error as e:
        # Roll back the transaction if an error occurs
        print(f"An error occurred: {e}")
        conn.execute("ROLLBACK")
    finally:
        if conn:
            conn.close()


# TODO: Evaluator

# --- Functions specific to this specific experiment --- #


def _C2_factors():
    """Fill Experiment Factors."""
    n_agent_factors = [4]
    policy_factors = ["default_het", "induced_hom"]
    agent_sensors = {
        "complete": None,
        "intersecting_span": {
            0: [1, 1, 1, 0, 0, 1, 1],
            1: [1, 0, 1, 1, 0, 1, 1],
            2: [1, 0, 0, 1, 1, 1, 1],
            3: [1, 1, 0, 0, 1, 1, 1],
        },
        "disjoint_span": {
            0: [1, 1, 0, 0, 0, 1, 1],
            1: [1, 0, 1, 0, 0, 1, 1],
            2: [1, 0, 0, 1, 0, 1, 1],
            3: [1, 0, 0, 0, 1, 1, 1],
        },
        "incomplete": {
            0: [1, 1, 0, 0, 0, 1, 1],
            1: [1, 0, 1, 0, 0, 1, 1],
            2: [1, 0, 0, 1, 0, 1, 1],
            3: [1, 0, 1, 0, 0, 1, 1],
        },
    }
    factors = list(product(n_agent_factors, policy_factors, agent_sensors))
    return [
        (a, p, json.dumps({"config": s, "agent_sensors": agent_sensors[s]}))
        for a, p, s in factors
    ]


def _rep_row_to_config(row):
    """ """
    config = {}
    config["num_agents"] = int(row["num_agents"])
    config["make_homo"] = row["policy_type"] == "induced_hom"
    sensors = json.loads(row["agent_sensors"])
    config["env_config"] = {
        "agent_sensors": {
            int(k): v for k, v in sensors["agent_sensors"].items()
        }
        if sensors["agent_sensors"]
        else None
    }
    config["save_dir"] = row["save_path"]
    return config


# --- Options for running as a script --- #


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
        const=30,
        default=0,
        help="Populate.",
    )
    parser.add_argument(
        "-C",
        "--clean",
        type=int,
        nargs="?",
        const=60,
        default=0,
        help="Clean records older than input minutes.",
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
    parser.add_argument(
        "-T",
        "--test",
        action="store_true",
        help="Run as a test.",
    )
    parser.add_argument(
        "--schema_path",
        type=Path,
        help="Path to schema default looks in CWD under the name `schema.sql`.",
    )
    args = parser.parse_args()
    main(args)
