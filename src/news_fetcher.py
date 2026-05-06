"""
Fetches news sentiment for F1 drivers and teams using NewsAPI + VADER.

Requires NEWSAPI_KEY environment variable (free key at newsapi.org, 500 req/day).
Falls back gracefully to 0.0 (neutral) if the key is missing or the request fails.

VADER sentiment: compound score in [-1.0, +1.0]
  - Positive: driver/team getting good press (wins, strong performance, upgrades)
  - Negative: bad press (crashes, reliability issues, controversy)
"""

import os
from datetime import datetime, timedelta
from typing import Optional

NEWSAPI_AVAILABLE = False
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    pass

VADER_AVAILABLE = False
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    VADER_AVAILABLE = True
except ImportError:
    pass


def _get_api_client() -> Optional[object]:
    key = os.environ.get("NEWSAPI_KEY", "")
    if not key or not NEWSAPI_AVAILABLE:
        return None
    return NewsApiClient(api_key=key)


def _vader_score(texts: list[str]) -> float:
    """Average VADER compound score across a list of text strings."""
    if not VADER_AVAILABLE or not texts:
        return 0.0
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(t)["compound"] for t in texts if t]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _fetch_articles(query: str, days_back: int = 7) -> list[str]:
    """
    Fetch recent news article titles + descriptions matching query.
    Returns list of strings (title + " " + description) for sentiment scoring.
    """
    client = _get_api_client()
    if client is None:
        return []

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        response = client.get_everything(
            q=query,
            from_param=from_date,
            sort_by="relevancy",
            language="en",
            page_size=20,
        )
        articles = response.get("articles", [])
        texts = []
        for article in articles:
            title = article.get("title") or ""
            description = article.get("description") or ""
            combined = f"{title} {description}".strip()
            if combined:
                texts.append(combined)
        return texts
    except Exception:
        return []


def fetch_driver_sentiment(driver_name: str, days_before: int = 7) -> float:
    """
    Fetch average news sentiment for a driver over the past `days_before` days.

    Args:
        driver_name: Full name (e.g., "Max Verstappen") or surname (e.g., "Verstappen")
        days_before: How many days of news to consider

    Returns:
        VADER compound score in [-1.0, +1.0], or 0.0 if unavailable
    """
    query = f'"{driver_name}" F1 Formula 1'
    texts = _fetch_articles(query, days_back=days_before)
    return _vader_score(texts)


def fetch_team_sentiment(team_name: str, days_before: int = 7) -> float:
    """
    Fetch average news sentiment for a team over the past `days_before` days.

    Args:
        team_name: Team name (e.g., "Red Bull Racing", "Ferrari", "Mercedes")
        days_before: How many days of news to consider

    Returns:
        VADER compound score in [-1.0, +1.0], or 0.0 if unavailable
    """
    query = f'"{team_name}" F1 Formula 1'
    texts = _fetch_articles(query, days_back=days_before)
    return _vader_score(texts)


def fetch_upgrade_flag(team_name: str, days_before: int = 14) -> bool:
    """
    Check whether a team has announced car upgrades recently.

    Searches for news articles combining the team name with upgrade-related keywords.
    Uses a wider default window (14 days) since upgrade news often appears earlier.

    Returns:
        True if upgrade-related articles were found, False otherwise
    """
    query = f'"{team_name}" F1 (upgrade OR "new parts" OR "development" OR "floor update" OR "aero package")'
    texts = _fetch_articles(query, days_back=days_before)
    return len(texts) > 0


def fetch_all_driver_sentiments(driver_names: list[str], days_before: int = 7) -> dict[str, float]:
    """
    Batch-fetch sentiment for multiple drivers.

    Returns dict mapping driver_name → sentiment score.
    """
    return {name: fetch_driver_sentiment(name, days_before) for name in driver_names}


def fetch_all_team_sentiments(team_names: list[str], days_before: int = 7) -> dict[str, float]:
    """
    Batch-fetch sentiment for multiple teams.

    Returns dict mapping team_name → sentiment score.
    """
    return {name: fetch_team_sentiment(name, days_before) for name in team_names}
