from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import re


def print_topics(H, words, num_top_words=20):
    for idx, topic in enumerate(H):
        top_words = [words[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
        print(f"Topic {idx+1}: {' '.join(top_words)}")


def topic_top_words(H, vocab, n=6):
    """Return list[list[str]] of top-n words for each topic (by weight)."""
    top_words = {}
    for topic_idx, topic in enumerate(H):
        top_indices = topic.argsort()[::-1][:n]  # highest weights
        top_terms = [vocab[i] for i in top_indices]
        top_words[topic_idx+1] = top_terms
    return top_words

def make_topic_labels(top_words, max_words=3, overrides=None):
    """
    Create short labels like 'bugs • patch • crash'.
    Pass `overrides={topic_id: "Your Label"}` for manual naming.
    """
    labels = {}
    for topic_idx, words in top_words.items():
        labels[topic_idx] = " / ".join(words[:max_words])
    return labels

def strip_reddit_preview(s: str) -> str:
    if not isinstance(s, str): return ""
    s = re.sub(r"\b(https?|www)\S+\b", " ", s)              # links
    s = re.sub(r"\b(redd|reddit|preview|width|height|png|jpg|jpeg|webp|auto|format)\b", " ", s)
    s = re.sub(r"\b(com|www|amp)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def nmf_topics(texts, k=8):
    vec = TfidfVectorizer(max_df=0.9, min_df=15, stop_words="english")
    X = vec.fit_transform(texts)
    model = NMF(n_components=k, random_state=42, init="nndsvda", max_iter=400)
    W = model.fit_transform(X)
    H = model.components_
    return model, vec, W, H