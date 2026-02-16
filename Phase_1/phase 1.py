# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:30:28 2026

@author: bruna
"""

# File: /mnt/data/Appliances_5.json  

import json
import re
import string
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

RANDOM_SEED = 42


# 1- Load Data 
DATA_PATH = "../data/Appliances_5.json" # r"C:\Users\bruna\Downloads\Appliances_5 (1).json\Appliances_5.json"

def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

df = load_jsonl(DATA_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(3))


# 2- Data Exploration
df.info()
missing = df.isna().mean().sort_values(ascending=False)
missing.head(15)

print(df["overall"].value_counts(dropna=False).sort_index())
df["overall"].describe()

if "verified" in df.columns:
    print("\n--- Verified Distribution ---")
    print(df["verified"].value_counts(dropna=False))

reviews_per_product = df["asin"].value_counts()
print(reviews_per_product.describe())
print("Top 10 most-reviewed products:\n", reviews_per_product.head(10))

reviews_per_user = df["reviewerID"].value_counts()
reviews_per_user.describe()
print("Top 10 most-active reviewers:\n", reviews_per_user.head(10))

def safe_text(x):
    return "" if pd.isna(x) else str(x)

df["reviewText"] = df.get("reviewText", "").apply(safe_text)
df["summary"] = df.get("summary", "").apply(safe_text)

df["review_len_chars"] = df["reviewText"].str.len()
df["review_len_words"] = df["reviewText"].apply(lambda t: len(t.split()) if t else 0)

print(df["review_len_chars"].describe())
print(df["review_len_words"].describe())

def iqr_outliers(series: pd.Series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return low, high, (series < low) | (series > high)

chars_low, chars_high, chars_is_out = iqr_outliers(df["review_len_chars"])
words_low, words_high, words_is_out = iqr_outliers(df["review_len_words"])

print(f"Chars outlier bounds: [{chars_low:.2f}, {chars_high:.2f}]  Outliers: {chars_is_out.sum()}")
print(f"Words outlier bounds: [{words_low:.2f}, {words_high:.2f}]  Outliers: {words_is_out.sum()}")

print("\nTop 5 longest reviews by words:")
print(df.nlargest(5, "review_len_words")[["asin", "overall", "review_len_words", "summary"]])

print("\nTop 5 shortest reviews by words:")
print(df.nsmallest(5, "review_len_words")[["asin", "overall", "review_len_words", "summary", "reviewText"]])

# Duplicates
# dup_all = df.duplicated(keep=False).sum() ERROR: TypeError: unhashable type: 'dict'
# print("Exact duplicate rows:", dup_all)

dup_all = df.duplicated(subset=['reviewerID','asin','reviewText'], keep=False).sum()
print("Number of duplicate reviews:", dup_all)





dup_key_cols = [c for c in ["reviewerID", "asin", "unixReviewTime", "reviewText"] if c in df.columns]
if len(dup_key_cols) >= 2:
    dup_key = df.duplicated(subset=dup_key_cols, keep=False).sum()
    print("Duplicates by key columns", dup_key_cols, ":", dup_key)

df_clean = df.drop_duplicates().copy()
print("After removing exact duplicates:", df_clean.shape)

plt.figure()
plt.hist(df_clean["overall"].dropna(), bins=np.arange(0.5, 6.6, 1))
plt.title("Rating Distribution (overall)")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.hist(reviews_per_product.values, bins=50)
plt.title("Distribution: #Reviews per Product (asin)")
plt.xlabel("#Reviews")
plt.ylabel("Number of products")
plt.yscale("log")  
plt.show()

plt.figure()
plt.hist(reviews_per_user.values, bins=50)
plt.title("Distribution: #Reviews per User (reviewerID)")
plt.xlabel("#Reviews")
plt.ylabel("Number of users")
plt.yscale("log")  
plt.show()

plt.figure()
plt.hist(df_clean["review_len_words"], bins=60)
plt.title("Distribution: Review Length (words)")
plt.xlabel("Words")
plt.ylabel("Count")
plt.show()


# 3-Basic Pre-processing / Labeling

#Label mapping
def rating_to_label(r):
    if pd.isna(r):
        return np.nan
    r = float(r)
    if r in [4.0, 5.0]:
        return "Positive"
    if r == 3.0:
        return "Neutral"
    if r in [1.0, 2.0]:
        return "Negative"
    return np.nan

df_clean["gold_label"] = df_clean["overall"].apply(rating_to_label)

df_clean["text_for_model"] = (df_clean["summary"].fillna("") + ". " + df_clean["reviewText"].fillna("")).str.strip()

df_clean = df_clean[df_clean["text_for_model"].str.len() > 0].copy()
df_clean = df_clean[df_clean["gold_label"].notna()].copy()

print(df_clean["gold_label"].value_counts())


# 4-Text Pre-processing 

URL_RE = re.compile(r"http\S+|www\.\S+")
MULTISPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")

def preprocess_for_vader(text: str) -> str:
    t = safe_text(text)
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

def preprocess_for_textblob(text: str) -> str:
    t = safe_text(text).lower()
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = re.sub(rf"[{re.escape(string.punctuation)}]{{4,}}", "!!!", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

df_clean["text_vader"] = df_clean["text_for_model"].apply(preprocess_for_vader)
df_clean["text_textblob"] = df_clean["text_for_model"].apply(preprocess_for_textblob)

df_clean["len_words_vader"] = df_clean["text_vader"].apply(lambda t: len(t.split()) if t else 0)
low_v, high_v, out_v = iqr_outliers(df_clean["len_words_vader"])
print("\nOutliers in words (VADER text) bounds:", (low_v, high_v), "count:", out_v.sum())


# 5- Using random to select 1000 reviews
sample_n = 1000
df_sample = df_clean.sample(n=sample_n, random_state=RANDOM_SEED).copy()
print("\nSample shape:", df_sample.shape)
print(df_sample["gold_label"].value_counts())


# 6- Modeling 
vader = SentimentIntensityAnalyzer()

def vader_predict_label(text: str, neutral_band: float = 0.05) -> str:
    scores = vader.polarity_scores(safe_text(text))
    c = scores["compound"]
    if c >= neutral_band:
        return "Positive"
    elif c <= -neutral_band:
        return "Negative"
    else:
        return "Neutral"

df_sample["pred_vader"] = df_sample["text_vader"].apply(vader_predict_label)

def textblob_predict_label(text: str, neutral_band: float = 0.05) -> str:
    polarity = TextBlob(safe_text(text)).sentiment.polarity
    if polarity >= neutral_band:
        return "Positive"
    elif polarity <= -neutral_band:
        return "Negative"
    else:
        return "Neutral"

df_sample["pred_textblob"] = df_sample["text_textblob"].apply(textblob_predict_label)


# 7-Validation + Comparison Table

LABELS = ["Negative", "Neutral", "Positive"]

def evaluate(y_true, y_pred, model_name: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "Model": model_name,
        "Accuracy": acc,
        "Macro_Precision": pr,
        "Macro_Recall": rc,
        "Macro_F1": f1
    }

y_true = df_sample["gold_label"].values

results = []
results.append(evaluate(y_true, df_sample["pred_vader"].values, "VADER"))
results.append(evaluate(y_true, df_sample["pred_textblob"].values, "TextBlob"))

comparison_table = pd.DataFrame(results).sort_values(by="Macro_F1", ascending=False)
print(comparison_table)

#reports
print(classification_report(y_true, df_sample["pred_vader"], labels=LABELS, zero_division=0))

print(confusion_matrix(y_true, df_sample["pred_vader"], labels=LABELS))

print(classification_report(y_true, df_sample["pred_textblob"], labels=LABELS, zero_division=0))

print(confusion_matrix(y_true, df_sample["pred_textblob"], labels=LABELS))


# 8-findings helpers

df_sample["vader_compound"] = df_sample["text_vader"].apply(lambda t: vader.polarity_scores(t)["compound"])
df_sample["textblob_polarity"] = df_sample["text_textblob"].apply(lambda t: TextBlob(t).sentiment.polarity)

print(df_sample.nlargest(5, "vader_compound")[["overall", "gold_label", "vader_compound", "summary"]])

print(df_sample.nsmallest(5, "vader_compound")[["overall", "gold_label", "vader_compound", "summary"]])

def simple_tokenize(text):
    t = safe_text(text).lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    tokens = [w for w in t.split() if w not in STOPWORDS and len(w) > 2]
    return tokens

pos_tokens = []
neg_tokens = []

for _, row in df_sample.iterrows():
    toks = simple_tokenize(row["text_for_model"])
    if row["gold_label"] == "Positive":
        pos_tokens.extend(toks)
    elif row["gold_label"] == "Negative":
        neg_tokens.extend(toks)

print(Counter(pos_tokens).most_common(20))

print(Counter(neg_tokens).most_common(20))


# 9- Save outputs 
OUT_CSV = "sample_1000_with_predictions.csv"
df_sample.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
