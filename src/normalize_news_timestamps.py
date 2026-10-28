
import pandas as pd

from db import DEFAULT_DB_PATH, get_connection


def main():
    with get_connection(DEFAULT_DB_PATH) as connection:
        rows = connection.execute("""
            SELECT id, published_at
            FROM news
            WHERE published_at IS NOT NULL
        """).fetchall()

        updates = []
        invalid = []

        for row in rows:
            original = row["published_at"]

            parsed = pd.to_datetime(
                original,
                errors="coerce",
                utc=True,
            )

            if pd.isna(parsed):
                invalid.append((row["id"], original))
                continue

            normalized = parsed.isoformat()

            if normalized != original:
                updates.append((normalized, row["id"]))

        print("News timestamps checked:", len(rows))
        print("Timestamps to normalize:", len(updates))
        print("Unparseable timestamps:", len(invalid))

        if invalid:
            print("\nExamples of unparseable timestamps:")
            for item in invalid[:10]:
                print(item)

        # Update only timestamps that parsed successfully.
        connection.executemany("""
            UPDATE news
            SET published_at = ?
            WHERE id = ?
        """, updates)

        print("\nTimestamps normalized:", len(updates))

        # Verify the chronological extremes after normalization.
        earliest = connection.execute("""
            SELECT published_at, title
            FROM news
            WHERE published_at IS NOT NULL
            ORDER BY julianday(published_at) ASC
            LIMIT 1
        """).fetchone()

        latest = connection.execute("""
            SELECT published_at, title
            FROM news
            WHERE published_at IS NOT NULL
            ORDER BY julianday(published_at) DESC
            LIMIT 1
        """).fetchone()

        print("\nEarliest parsed timestamp:")
        print(dict(earliest) if earliest else None)

        print("\nLatest parsed timestamp:")
        print(dict(latest) if latest else None)


if __name__ == "__main__":
    main()
