from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if not isinstance(text, str):
        return 0.0
    return analyzer.polarity_scores(text)["compound"]

def label_sentiment(s):
    if s > 0.05:  return "positive"
    if s < -0.05: return "negative"
    return "neutral"

def label_score(s):
    if s > 0.05:  return "mostly positive"
    if s < -0.05: return "mostly negative"
    return "mostly neutral"

def safe_weighted_mean(vals, wts):
    # This wont give me a issue with weights that sum to 0 
    v = np.array(vals, float)
    w = np.array(wts, float)
    return np.average(v, weights=w) if w.sum() > 0 else np.nanmean(v)

