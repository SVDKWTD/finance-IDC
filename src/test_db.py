
import tempfile
from pathlib import Path

from db import (
    create_tables,
    upsert_market_price,
    upsert_news,
    read_market_prices,
    read_news,
    update_news_title,
)


def main():
    # Create a temporary database for testing.
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db = Path(temp_dir) / "test_market.db"

        create_tables(test_db)

        # Test inserting a market-price record.
        price = {
            "date": "2026-10-23",
            "open": 25000,
            "high": 25200,
            "low": 24900,
            "close": 25100,
            "volume": 100000,
        }

        upsert_market_price("^NSEI", price, test_db)
        upsert_market_price("^NSEI", price, test_db)

        prices = read_market_prices(test_db)
        assert len(prices) == 1, "Duplicate price record was created."

        # Test inserting a news article.
        article = {
            "title": "Example financial news headline",
            "published": "2026-10-23T10:00:00Z",
            "source": "Test source",
            "link": "https://example.com/test-article",
            "collected_at": "2026-10-23T10:05:00Z",
        }

        upsert_news(article, test_db)
        upsert_news(article, test_db)

        news = read_news(test_db)
        assert len(news) == 1, "Duplicate news record was created."

        # Test updating an existing news title.
        updated = update_news_title(
            article["link"],
            "Updated example headline",
            test_db,
        )

        assert updated == 1, "News title was not updated."
        assert read_news(test_db)[0]["title"] == "Updated example headline"

        print("All SQLite tests passed.")
        print("Price records:", len(read_market_prices(test_db)))
        print("News records:", len(read_news(test_db)))


if __name__ == "__main__":
    main()
