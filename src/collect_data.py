import praw
import pandas as pd
import os
from dotenv import load_dotenv
from typing import List, Dict
import time
from datetime import datetime
import numpy as np
import sqlite3
from typing import Iterable

# --- Load environment variables ---
load_dotenv()

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID") or os.getenv("REDDIT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET") or os.getenv("REDDIT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# --- SQLite paths and connection ---
from pathlib import Path
DB_DIR = Path("../data/db")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "sims4.db"

def get_conn(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn



__all__ = [
    "collect_reddit_posts",
    "collect_comments_for_posts",
    "init_db",
    "get_conn",
    "ensure_posts_schema",
    "ensure_comments_schema",
    "insert_posts",
    "insert_comments",
    "get_existing_ids",
    "safe_concat",
    "harvest_posts",
    "insert_posts_df",
    "get_comment_candidates",
    "collect_comments_for_candidates",
]






# --- getting posts ---
def _get_reddit(client_id: str | None = None, client_secret: str | None = None, user_agent: str | None = None):
    """
    Internal helper: return a configured PRAW Reddit instance using provided creds
    or falling back to environment variables.
    """
    cid = client_id or CLIENT_ID
    csec = client_secret or CLIENT_SECRET
    ua = user_agent or USER_AGENT
    if not (cid and csec and ua):
        raise ValueError("Reddit credentials missing. Provide client_id/client_secret/user_agent or set them in .env")
    return praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)



def collect_reddit_posts(
    client_id: str | None = None,
    client_secret: str | None = None,
    user_agent: str | None = None,
    subreddit_name: str | list[str] = "Sims4",
    limit: int = 500,
    time_filter: str = "year", 
    mode: str | list[str] = "hot" 
) -> pd.DataFrame:
    """
    Collect posts from one or more subreddits and one or more modes.

    Args:
        client_id/client_secret/user_agent: Reddit API credentials (optional if set in .env)
        subreddit_name: a single subreddit (str) or list of subreddit names (e.g., ["Sims4","thesims"])
        limit: max posts per subreddit per mode
        time_filter: only used for mode="top"
        mode: a single mode or a list of modes among {"top","new","hot"}

    Returns:
        DataFrame with columns:
            id, created_utc, created_date, author, title, body, score, num_comments, permalink, subreddit, mode
    """
    reddit = _get_reddit(client_id, client_secret, user_agent)

    # normalize params
    subs = [subreddit_name] if isinstance(subreddit_name, str) else list(subreddit_name)
    modes = [mode] if isinstance(mode, str) else list(mode)

    rows: List[Dict] = []
    for sub in subs:
        sr = reddit.subreddit(sub)
        for m in modes:
            if m not in {"top", "new", "hot"}:
                raise ValueError("mode must be one or more of {'top','new','hot'}")
            if m == "top":
                stream = sr.top(limit=limit, time_filter=time_filter)
            elif m == "new":
                stream = sr.new(limit=limit)
            else:  # "hot"
                stream = sr.hot(limit=limit)

            for post in stream:
                created = getattr(post, "created_utc", None)
                rows.append({
                    "id": post.id,
                    "created_utc": created,
                    "created_date": (datetime.utcfromtimestamp(created) if created else None),
                    "author": str(getattr(post.author, "name", None)),
                    "title": post.title or "",
                    "body": post.selftext or "",
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "permalink": f"https://reddit.com{post.permalink}",
                    "subreddit": sub,
                    "mode": m,
                })

    return pd.DataFrame(rows)








# --- getting comments ---
def collect_comments_for_posts(client_id: str | None, client_secret: str | None, user_agent: str | None, post_ids: List[str], sleep: float = 0.4) -> pd.DataFrame:
    reddit = _get_reddit(client_id, client_secret, user_agent)
    rows = []
    for pid in post_ids:
        try:
            submission = reddit.submission(id=pid)
            submission.comments.replace_more(limit=0)
            for c in submission.comments.list():
                rows.append({
                    "post_id": pid,
                    "comment_id": c.id,
                    "created_utc": getattr(c, "created_utc", None),
                    "author": str(getattr(c.author, "name", None)),
                    "body": c.body or "",
                    "score": c.score,
                    "parent_permalink": f"https://reddit.com{submission.permalink}"
                })
            time.sleep(sleep)  
        except Exception as e:
            print(f"Skipping {pid}: {e}")
            time.sleep(1)
    return pd.DataFrame(rows)







# --- Making relational database ---

POSTS_DDL = """
CREATE TABLE IF NOT EXISTS posts (
    post_id TEXT PRIMARY KEY,
    created_utc INTEGER,
    date TEXT,
    author TEXT,
    title TEXT,
    body TEXT,
    score INTEGER,
    num_comments INTEGER,
    permalink TEXT,
    subreddit TEXT,
    mode TEXT
);
"""

COMMENTS_DDL = """
CREATE TABLE IF NOT EXISTS comments (
    comment_id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL,
    subreddit TEXT,
    created_utc INTEGER,
    date TEXT,
    author TEXT,
    body TEXT,
    score INTEGER,
    parent_permalink TEXT,
    FOREIGN KEY (post_id) REFERENCES posts(post_id) ON DELETE CASCADE
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit);",
    "CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_utc);",
    "CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);",
    "CREATE INDEX IF NOT EXISTS idx_comments_created ON comments(created_utc);"
]

def init_db():
    with get_conn() as conn:
        conn.execute(POSTS_DDL)
        conn.execute(COMMENTS_DDL)
        for ddl in INDEXES:
            conn.execute(ddl)


def ensure_posts_schema(df_posts: pd.DataFrame) -> pd.DataFrame:
    df = df_posts.copy()

    # rename id -> post_id, created_date -> date (if present)
    rename_map = {}
    if "id" in df.columns: rename_map["id"] = "post_id"
    if "created_date" in df.columns: rename_map["created_date"] = "date"
    df = df.rename(columns=rename_map)

    # create 'date' if missing (ISO-8601 UTC)
    if "date" not in df.columns:
        if "created_utc" in df.columns and df["created_utc"].notna().any():
            df["date"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            df["date"] = pd.NaT

    # enforce column order
    cols = ["post_id","created_utc","date","author","title","body","score","num_comments","permalink","subreddit","mode"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[cols].copy()

    # types (SQLite is flexible, but we coerce sensibly)
    for int_col in ["created_utc","score","num_comments"]:
        df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")
    for txt in ["post_id","author","title","body","permalink","subreddit","mode","date"]:
        df[txt] = df[txt].astype("string")

    # drop rows missing primary key
    df = df[df["post_id"].notna()]
    return df



def ensure_comments_schema(df_comments: pd.DataFrame, df_posts_for_subs: pd.DataFrame) -> pd.DataFrame:
    df = df_comments.copy()

    # ensure subreddit for comments (derive from posts if missing)
    if "subreddit" not in df.columns or df["subreddit"].isna().all():
        sub_map = df_posts_for_subs.set_index("post_id")["subreddit"].to_dict() if "post_id" in df_posts_for_subs.columns else \
                  df_posts_for_subs.set_index("id")["subreddit"].to_dict()
        df["subreddit"] = df["post_id"].map(sub_map)

    # create 'date' if missing
    if "date" not in df.columns:
        if "created_utc" in df.columns and df["created_utc"].notna().any():
            df["date"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            df["date"] = pd.NaT

    # enforce column order
    cols = ["post_id","comment_id","subreddit","created_utc","date","author","body","score","parent_permalink"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols].copy()

    # types
    for int_col in ["created_utc","score"]:
        df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")
    for txt in ["post_id","comment_id","subreddit","author","body","parent_permalink","date"]:
        df[txt] = df[txt].astype("string")

    # drop rows missing PK or FK
    df = df[df["comment_id"].notna() & df["post_id"].notna()]
    return df




POSTS_INSERT = """
INSERT OR IGNORE INTO posts
(post_id, created_utc, date, author, title, body, score, num_comments, permalink, subreddit, mode)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

COMMENTS_INSERT = """
INSERT OR IGNORE INTO comments
(comment_id, post_id, subreddit, created_utc, date, author, body, score, parent_permalink)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

def chunk_iter(iterable: Iterable, size: int = 1000):
    buf = []
    for row in iterable:
        buf.append(row)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

def insert_posts(conn, df_posts_norm: pd.DataFrame, batch_size: int = 1000):
    df = df_posts_norm.copy()
    # convert pandas NA/NaN to Python None so sqlite can bind them
    df = df.astype(object).where(pd.notnull(df), None)

    tuples = df[[
        "post_id","created_utc","date","author","title","body",
        "score","num_comments","permalink","subreddit","mode"
    ]].itertuples(index=False, name=None)

    cur = conn.cursor()
    for batch in chunk_iter(tuples, batch_size):
        cur.executemany(POSTS_INSERT, batch)
    conn.commit()

def insert_comments(conn, df_comments_norm: pd.DataFrame, batch_size: int = 1000):
    df = df_comments_norm.copy()
    # same NA -> None conversion
    df = df.astype(object).where(pd.notnull(df), None)

    tuples = df[[
        "comment_id","post_id","subreddit","created_utc","date",
        "author","body","score","parent_permalink"
    ]].itertuples(index=False, name=None)

    cur = conn.cursor()
    for batch in chunk_iter(tuples, batch_size):
        cur.executemany(COMMENTS_INSERT, batch)
    conn.commit()
    
# ---------------------------
# Scaling helpers (harvest + upsert)
# ---------------------------

def get_existing_ids(table: str, id_col: str) -> set:
    """Return a set of existing primary keys from a table."""
    import pandas as pd
    with get_conn() as conn:
        q = f"SELECT {id_col} FROM {table};"
        df = pd.read_sql_query(q, conn)
    return set(df[id_col].tolist())

def safe_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Concat a list of DataFrames, skipping None/empty frames."""
    import pandas as pd
    keep = [d for d in dfs if d is not None and not d.empty]
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()

def harvest_posts(
    subs: List[str],
    modes: List[str] = ("top", "hot", "new"),
    time_filters: List[str] = ("month", "year", "all"),
    limit_per: int = 1000,
    sleep_between: float = 0.4
) -> pd.DataFrame:
    """
    Collect posts across multiple (sub, mode, [time_filter]) combos and
    return a de-duplicated DataFrame excluding rows already present in DB.
    """
    existing_ids = get_existing_ids("posts", "post_id")
    collected: List[pd.DataFrame] = []

    for sub in subs:
        for mode in modes:
            if mode == "top":
                for tf in time_filters:
                    df = collect_reddit_posts(
                        subreddit_name=sub,
                        mode=mode,
                        time_filter=tf,
                        limit=limit_per
                    )
                    if not df.empty:
                        df = df[~df["id"].isin(existing_ids)]
                        collected.append(df)
                    time.sleep(sleep_between)
            else:
                df = collect_reddit_posts(
                    subreddit_name=sub,
                    mode=mode,
                    limit=limit_per
                )
                if not df.empty:
                    df = df[~df["id"].isin(existing_ids)]
                    collected.append(df)
                time.sleep(sleep_between)

    return safe_concat(collected)

def insert_posts_df(df_posts: pd.DataFrame, batch_size: int = 2000) -> int:
    """Normalize then insert posts; returns number of inserted rows (attempted)."""
    if df_posts is None or df_posts.empty:
        return 0
    df_norm = ensure_posts_schema(df_posts).drop_duplicates(subset=["post_id"])
    with get_conn() as conn:
        insert_posts(conn, df_norm, batch_size=batch_size)
    return len(df_norm)

def get_comment_candidates(limit: int = 5000) -> List[str]:
    """
    Return post_ids that currently have zero comments stored in the DB,
    prioritizing high-score posts.
    """
    import pandas as pd
    with get_conn() as conn:
        q = f"""
            SELECT p.post_id
            FROM posts p
            LEFT JOIN (
                SELECT post_id, COUNT(*) AS c
                FROM comments
                GROUP BY post_id
            ) cc ON cc.post_id = p.post_id
            WHERE IFNULL(cc.c,0) = 0
            ORDER BY p.score DESC
            LIMIT {int(limit)};
        """
        return pd.read_sql_query(q, conn)["post_id"].tolist()

def collect_comments_for_candidates(
    post_ids: List[str],
    batch_posts: int = 200,
    sleep_between: float = 0.4,
    insert_batch_size: int = 2000
) -> int:
    """
    Fetch comments for the provided post_ids in batches, normalize, and insert.
    Returns total number of comments inserted (attempted).
    """
    total_inserted = 0

    def _chunks(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i+size]

    for batch in _chunks(post_ids, batch_posts):
        df_c = collect_comments_for_posts(None, None, None, batch, sleep=sleep_between)
        if df_c is None or df_c.empty:
            continue
        # We pass df_posts placeholder to ensure_comments_schema for subreddit mapping if needed
        df_c_norm = ensure_comments_schema(df_c, pd.DataFrame(columns=["post_id","subreddit"]))
        df_c_norm = df_c_norm.drop_duplicates(subset=["comment_id"])
        with get_conn() as conn:
            insert_comments(conn, df_c_norm, batch_size=insert_batch_size)
        total_inserted += len(df_c_norm)

    return total_inserted