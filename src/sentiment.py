from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
SARC_MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"

sent_tok = AutoTokenizer.from_pretrained(SENT_MODEL_NAME)
sent_mdl = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_NAME).to(DEVICE).eval()

sarc_tok = AutoTokenizer.from_pretrained(SARC_MODEL_NAME)
sarc_mdl = AutoModelForSequenceClassification.from_pretrained(SARC_MODEL_NAME).to(DEVICE).eval()

# Data Cleaning

# --- precompiled patterns (fast + reusable)
_USERNAME   = re.compile(r"(?<!\w)u\/[A-Za-z0-9_-]+|\@[A-Za-z0-9_]+")
_URL        = re.compile(r"https?://\S+|www\.\S+")
_MDQUOTE    = re.compile(r"(^|\n)\s{0,3}>\s.*")   # blockquotes like "> text"
_WHITESPACE = re.compile(r"\s+")

# Target preview blobs more precisely (domains/params), not normal words.
_PREVIEW_DOMAINS = re.compile(
    r"(https?://)?("
    r"(preview\.redd\.it|i\.redd\.it|v\.redd\.it|redditstatic\.com)"
    r"|(\S+\.(?:png|jpe?g|webp|gif|mp4))(?:\?\S+)?"
    r")",
    re.IGNORECASE
)

# Common preview query params (width/height/format/autoplay/etc.)
_PREVIEW_PARAMS = re.compile(
    r"(\b(w|width|h|height|format|auto|fit|crop|quality|q|s|amp|fps|duration|loop|mute|controls)"
    r"=\S+)",
    re.IGNORECASE
)

def preprocess(text: str, *, remove_preview: bool = True) -> str:
    """
    Unified cleaner for Reddit/Forums text.
    - Normalizes usernames/links/quotes/HTML entities.
    - Optionally strips Reddit preview links and their noisy query params.
    - Leaves ordinary words like 'reddit' intact unless part of a URL.

    Args:
        text: raw post/comment string
        remove_preview: if True, removes preview URLs and noisy media params

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    t = text

    # 1) Optional: drop preview/media URLs and their query gunk
    if remove_preview:
        # Remove preview/media URLs (keeps surrounding text)
        t = _PREVIEW_DOMAINS.sub(" ", t)
        # Remove query-like param shrapnel that often remains after URL splits
        t = _PREVIEW_PARAMS.sub(" ", t)

    # 2) Normalize obvious social/markdown noise
    t = _USERNAME.sub("@user", t)       # collapse u/handles and @mentions
    t = _URL.sub("http", t)             # collapse any remaining links to 'http'
    t = _MDQUOTE.sub("", t)             # drop quoted blocks
    t = t.replace("&amp;", "&")         # HTML entity -> char

    # 3) Whitespace tidy
    t = _WHITESPACE.sub(" ", t).strip()

    return t

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



# --- instead of running all at the same time, this helps with memory usage
def batched(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i+batch_size]
        
"""
========================== Sentiment Analysis ==========================
"""

# --- sentiment prediction
@torch.inference_mode()
def predict_sentiment(texts, batch_size=256):
    """Return probs (N x 3) in [neg, neu, pos] order."""
    all_probs = []
    for batch in batched(texts, batch_size):
        enc = sent_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = sent_mdl(**enc).logits
        all_probs.append(softmax(logits.detach().cpu().numpy(), axis=1))
    return np.vstack(all_probs)  # shape: (N, 3)


# --- sarcasm prediction
@torch.inference_mode()
def predict_sarcasm(texts, batch_size=256):
    """Return sarcasm probability (irony) as a scalar in [0,1]."""
    probs_ironies = []
    for batch in batched(texts, batch_size):
        enc = sarc_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = sarc_mdl(**enc).logits.detach().cpu().numpy()  # shape: (B, 2)
        p = softmax(logits, axis=1)[:, 1]  # prob of 'irony'
        probs_ironies.append(p)
    return np.concatenate(probs_ironies)  # shape: (N,)


# --- sarcasm-aware scoring
# If sarcastic, the language often flips polarity; we downweight raw sentiment
# and gently bias toward negativity (tunable).
def effective_negativity(p_sent_3, p_sarcasm, downweight=0.35, sarcasm_bias=0.10):
    """
    p_sent_3: (N,3) probs in [neg, neu, pos]
    p_sarcasm: (N,) prob of irony
    Returns:
      neg_raw:  P(neg) - P(pos)
      neg_eff:  sarcasm-aware effective negativity
    """
    neg = p_sent_3[:, 0]
    pos = p_sent_3[:, 2]
    neg_raw = neg - pos
    # interpolate: if sarcasm high, trust neg less (it could be ironic praise),
    # but add a small negative bias because sarcastic praise usually implies criticism.
    neg_eff = (1 - p_sarcasm) * neg_raw + p_sarcasm * (downweight * neg_raw + sarcasm_bias)
    return neg_raw, neg_eff



# --- gives me the entire analysis in one
def score_texts(texts, batch_size=256, preprocess_fn=preprocess):
    texts_prep = [preprocess_fn(t) for t in texts]
    p_sent = predict_sentiment(texts_prep, batch_size=batch_size)     # (N,3)
    p_sarc = predict_sarcasm(texts_prep, batch_size=batch_size)       # (N,)
    neg_raw, neg_eff = effective_negativity(p_sent, p_sarc)
    # convenient outputs
    result = {
        "neg_prob": p_sent[:, 0],
        "neu_prob": p_sent[:, 1],
        "pos_prob": p_sent[:, 2],
        "sarcasm_prob": p_sarc,
        "neg_raw": neg_raw,     # P(neg) - P(pos)
        "neg_effective": neg_eff
    }
    return result

