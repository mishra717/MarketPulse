"""Headline sentiment using VADER (rule-based, no model download needed)."""
from __future__ import annotations


def score_headlines(headlines: list[str]) -> list[tuple[str, float]]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return [(h, analyzer.polarity_scores(h)["compound"]) for h in headlines]


def average_sentiment(scored: list[tuple[str, float]]) -> float:
    return sum(s for _, s in scored) / len(scored) if scored else 0.0


def label(score: float) -> str:
    return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
