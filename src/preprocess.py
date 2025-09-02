
import pandas as pd
from datetime import datetime

def basic_clean(df, text_cols=("title","body")):
    df = df.copy()
    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
    # drop exact dupes on id or text
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    else:
        # fallback: dedupe by text
        keep_cols = [c for c in df.columns if c in text_cols]
        if keep_cols:
            df = df.drop_duplicates(subset=keep_cols)
    return df


def collect_reddit_posts(
    client_id,
    client_secret,
    user_agent,
    subreddit_name="Sims4",
    limit=500,
    time_filter="year",
    mode="top"
):
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    sr = reddit.subreddit(subreddit_name)

    if mode == "top":
        stream = sr.top(limit=limit, time_filter=time_filter)
    elif mode == "new":
        stream = sr.new(limit=limit)
    elif mode == "hot":
        stream = sr.hot(limit=limit)
    else:
        raise ValueError("mode must be one of {'top','new','hot'}")

    rows = []
    for post in stream:
        rows.append({
            "id": post.id,
            "created_utc": post.created_utc,  # raw unix timestamp
            "created_date": datetime.utcfromtimestamp(post.created_utc),  # human-readable
            "author": str(getattr(post.author, "name", None)),
            "title": post.title or "",
            "body": post.selftext or "",
            "score": post.score,
            "num_comments": post.num_comments,
            "permalink": f"https://reddit.com{post.permalink}",
            "subreddit": subreddit_name
        })
    return pd.DataFrame(rows)