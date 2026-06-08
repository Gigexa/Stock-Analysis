import json
from pathlib import Path
import anaylser
import feedparser
import joblib
import numpy as np
import pandas as pd
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NewsStockMatcher:

    def __init__(
        self,
        stock_csv="all_enriched.csv",
        rss_url="https://news.ycombinator.com/rss",
        processed_file="processed_news.json",
        vectorizer_file="vectorizer.pkl",
        stock_vectors_file="stock_vectors.pkl",
        top_k=10,
        similarity_threshold=0.05,
    ):

        self.stock_csv = stock_csv
        self.rss_url = rss_url
        self.processed_file = processed_file

        self.vectorizer_file = vectorizer_file
        self.stock_vectors_file = stock_vectors_file

        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50000,
        )

        self.data = None
        self.stock_vectors = None

        self.processed = self.load_processed_news()

    # --------------------------------------------------
    # Processed news handling
    # --------------------------------------------------

    def load_processed_news(self):

        if Path(self.processed_file).exists():

            with open(self.processed_file, "r") as f:
                return set(json.load(f))

        return set()

    def save_processed_news(self):

        with open(self.processed_file, "w") as f:
            json.dump(list(self.processed), f)

    # --------------------------------------------------
    # Stocks
    # --------------------------------------------------

    def load_stocks(self):

        self.data = pd.read_csv(self.stock_csv)

        self.data["enhanced_description"] = (
            self.data["enhanced_description"]
            .fillna("")
            .astype(str)
        )

    def build_stock_texts(self):

        stock_texts = []

        for _, row in self.data.iterrows():

            text = f"""
            {row.get('industry', '')}
            {row.get('name', '')}
            {row.get('enhanced_description', '')}
            """

            stock_texts.append(text)

        return stock_texts

    def build_or_load_stock_vectors(self):

        vectorizer_exists = Path(
            self.vectorizer_file
        ).exists()

        vectors_exist = Path(
            self.stock_vectors_file
        ).exists()

        if vectorizer_exists and vectors_exist:

            print("Loading cached vectors...")

            self.vectorizer = joblib.load(
                self.vectorizer_file
            )

            self.stock_vectors = joblib.load(
                self.stock_vectors_file
            )

            return

        print("Building stock vectors...")

        stock_texts = self.build_stock_texts()

        self.stock_vectors = (
            self.vectorizer.fit_transform(stock_texts)
        )

        joblib.dump(
            self.vectorizer,
            self.vectorizer_file
        )

        joblib.dump(
            self.stock_vectors,
            self.stock_vectors_file
        )

    # --------------------------------------------------
    # News
    # --------------------------------------------------

    def fetch_news(self):

        feed = feedparser.parse(self.rss_url)

        articles = []

        for entry in feed.entries:

            news_id = getattr(
                entry,
                "link",
                entry.title
            )

            if news_id in self.processed:
                continue

            articles.append(
                {
                    "id": news_id,
                    "title": entry.title,
                    "link": entry.link,
                }
            )

        return articles

    def get_article_text(self, url):

        try:

            article = Article(url)

            article.download()
            article.parse()

            text = article.text.strip()

            if len(text) < 200:
                return None

            return text

        except Exception:

            return None

    def build_news_texts(self, news_articles):

        news_texts = []

        valid_articles = []

        for article in news_articles:

            text = self.get_article_text(
                article["link"]
            )

            if text is None:

                text = article["title"]

            news_texts.append(text)
            valid_articles.append(article)

        return valid_articles, news_texts

    # --------------------------------------------------
    # Matching
    # --------------------------------------------------

    def find_matches(
        self,
        news_articles,
        news_vectors,
    ):

        results = []

        for idx, article in enumerate(
            news_articles
        ):

            scores = cosine_similarity(
                news_vectors[idx],
                self.stock_vectors,
            )[0]

            top_indices = np.argsort(
                scores
            )[-self.top_k:][::-1]

            matches = []

            for stock_idx in top_indices:

                score = scores[stock_idx]

                if score < self.similarity_threshold:
                    continue

                stock = self.data.iloc[
                    stock_idx
                ]

                matches.append(
                    {
                        "symbol": stock["symbol"],
                        "name": stock["name"],
                        "score": float(score),
                    }
                )

            results.append(
                {
                    "news": article["title"],
                    "matches": matches,
                }
            )

        return results

    # --------------------------------------------------
    # Processed marking
    # --------------------------------------------------

    def mark_processed(self, news_articles):

        for article in news_articles:

            self.processed.add(
                article["id"]
            )

        self.save_processed_news()

    # --------------------------------------------------
    # Main
    # --------------------------------------------------

    def run(self):

        self.load_stocks()

        self.build_or_load_stock_vectors()

        news_articles = self.fetch_news()

        if not news_articles:

            print("No new articles.")
            return

        news_articles, news_texts = (
            self.build_news_texts(
                news_articles
            )
        )

        if not news_texts:

            print("No valid news.")
            return

        news_vectors = (
            self.vectorizer.transform(
                news_texts
            )
        )

        matches = self.find_matches(
            news_articles,
            news_vectors,
        )

        signals = set()

        for result in matches:

            print()
            print("=" * 80)
            print("NEWS:")
            print(result["news"])

            print()
            print("MATCHES:")

            for match in result["matches"]:

                print(
                    match["symbol"],
                    match["name"],
                    round(
                        match["score"],
                        3
                    ),
                )

                signals.add(
                    match["symbol"]
                )

        with open(
            "signals_to_analyse.txt",
            "a",
            encoding="utf-8",
        ) as f:

            for symbol in sorted(signals):

                analysis = anaylser.Analyzer(symbol, period = '5d', interval = '1h')
                analysis.analyze()


        self.mark_processed(
            news_articles
        )


if __name__ == "__main__":

    matcher = NewsStockMatcher(
        stock_csv="all_enriched.csv",
        top_k=10,
        similarity_threshold=0.05,
    )

    matcher.run()