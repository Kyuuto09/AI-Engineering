import csv
import json
import os
import re
from collections import Counter

import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

# https://www.trustpilot.com/review/jerseywatch.com


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def sentiment_label(polarity):
    if polarity > 0.05:
        return "positive"
    if polarity < -0.05:
        return "negative"
    return "neutral"


def main():
    # Download required NLTK resources (safe to call multiple times)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "..", "data", "customers_review.csv")

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    processed = []
    sentiment_counts = Counter()

    with open(data_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            title = (row.get("Review Title") or "").strip()
            content = (row.get("Review Content") or "").strip()
            text = (title + ". " + content).strip(" .")

            if not text:
                continue

            # Tokenization
            tokens = tokenize(text)
            # Stop-word removal
            tokens_no_stop = [t for t in tokens if t not in stop_words]
            # Stemming
            stems = [stemmer.stem(t) for t in tokens_no_stop]
            # Lemmatization
            lemmas = [lemmatizer.lemmatize(t) for t in tokens_no_stop]

            polarity = TextBlob(text).sentiment.polarity
            label = sentiment_label(polarity)
            sentiment_counts[label] += 1

            processed.append(
                {
                    "position": row.get("Position"),
                    "reviewer": row.get("Reviewer Name"),
                    "review_date": row.get("Review Date"),
                    "review_title": title,
                    "review_content": content,
                    "rating_description": row.get("Rating Description"),
                    "tokens": tokens,
                    "tokens_no_stopwords": tokens_no_stop,
                    "stems": stems,
                    "lemmas": lemmas,
                    "polarity": polarity,
                    "sentiment": label,
                }
            )

    output_path = os.path.join(base_dir, "processed_reviews.json")
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(processed, json_file, ensure_ascii=True, indent=2)

    print("Sentiment counts:", dict(sentiment_counts))
    print(f"Saved JSON: {output_path}")

    # Plot sentiment counts
    labels = ["negative", "neutral", "positive"]
    counts = [sentiment_counts.get(label, 0) for label in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color=["#d9534f", "#f0ad4e", "#5cb85c"])
    plt.title("Trust pilot: jerseywatch.com")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig("review_graph_jerseywatch.com.png")
    plt.show()


if __name__ == "__main__":
    main()
