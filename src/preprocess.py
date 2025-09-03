import pandas as pd
from datetime import datetime
import re
import spacy
from nltk.corpus import stopwords
import sqlite3

try:
    nlp = spacy.load("en_core_web_md")
except Exception:
    nlp = None

try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = {"the", "and", "is", "in", "to", "of"}



"""
Function recieves: a text string
It then lowercases, removes URLs and mentions, and applies lemmatization.
"""
def clean_text_value(text, keep_numbers=True, remove_urls=True) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    if remove_urls:
        text = re.sub(r"http\S+|www\S+|@\w+", "", text)     # remove urls and mentions
    if keep_numbers:
        text = re.sub(r"[^a-z0-9\s]", "", text)             # keep letters and numbers
    else:
        text = re.sub(r"[^a-z\s]", "", text)                # keep only letters
    if nlp:
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_space]
        return " ".join(tokens)
    else:
        tokens = [word for word in text.split() if word not in stop_words]
        return " ".join(tokens)



"""
This function applies the clean_text_value function to a pandas Series.
I will be using this on my DataFrame columns.
"""
def clean_text_series(series: pd.Series, **kwargs) -> pd.Series:
    return series.fillna("").map(lambda x: clean_text_value(x, **kwargs))



"""
This function tokenizes a text string into individual words.
"""
def tokenize_series(series: pd.Series, **kwargs) -> pd.Series:
    def tokenize(text):
        cleaned = clean_text_value(text, **kwargs)
        return cleaned.split()
    return series.fillna("").map(tokenize)




"""
This function fixes byte sequences in a pandas Series.
Thats because when you display the dif, you want to see the actual integer values instead of byte sequences.
"""
def fix_bytes_series(series: pd.Series) -> pd.Series:
    return series.map(lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, (bytes, bytearray)) else x)



"""
This function converts a pandas Series of Unix timestamps to datetime objects.
This way our date column can be read
"""
def created_utc_to_date_series(series: pd.Series, unit="s") -> pd.Series:
    def convert(val):
        if isinstance(val, (bytes, bytearray)):
            val = int.from_bytes(val, byteorder="little")
        try:
            return pd.to_datetime(val, unit=unit, utc=True)
        except Exception:
            return pd.NaT
    return series.map(convert)



"""
Since I made a list in the pandas df of all of the tokenized columns, I need to serialize them before saving to the database.
The best practice would be to make a new relationship in the database that connects every post and comment to its respective tokens, but that would create ~20M rows and would take hours to complete. Because of that, I am chosing to go down this route. 
"""
def serialize_list_columns(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            # If any cell in this column is a list, join into a space-separated string
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(
                    lambda x: " ".join(x) if isinstance(x, list) else x
                )
    return df



