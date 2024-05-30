import httpx
import sqlite3
import asyncio
import datetime
from rich import print
from time import perf_counter
from typing import List
from transformers import pipeline
from dataclasses import dataclass
from selectolax.parser import HTMLParser


@dataclass
class Page:
    url: str
    html: str


@dataclass
class Article:
    date: str
    url: str
    summary: str


@dataclass
class Sentiment:
    date: str
    time: str
    url: str
    summary: str
    sentiment: str
    score: float


async def get_pages(pages: int) -> List[Page]:
    all_pages: List[Page] = []
    async with httpx.AsyncClient() as client:
        tasks = []
        for page in range(1, pages +1):
            url = f"https://oilprice.com/Latest-Energy-News/World-News/Page-{page}.html"
            tasks.append(client.get(url))
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            page = Page(url=resp.request.url, html=resp.text)
            all_pages.append(page)
    return all_pages


def parse_pages(all_pages: List[Page]) -> List[Article]:
    all_articles: List[Article] = []
    for page in all_pages:
        html = HTMLParser(page.html)
        articles: List[str] = html.css("div.categoryArticle__content")
        for article in articles:
            url: str = article.css_first("a").attrs["href"]
            date: str = article.css_first("p.categoryArticle__meta").text().split("|")[0].strip()
            summary: str = article.css_first("p.categoryArticle__excerpt").text()
            summary = summary.replace("…","").replace(".", "").replace("\xa0", " ").replace("\n", " ")
            article = Article(date=date, url=url, summary=summary)
            all_articles.append(article)
    return all_articles


def get_sentiment(all_articles: List[Article]) -> List[Sentiment]:
    all_sentiment: List[Sentiment] = []
    pipe = pipeline("sentiment-analysis",
        model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    )
    for article in all_articles:
        trans = pipe(article.summary)
        dt = datetime.datetime.strptime(article.date, "%B %d, %Y at %H:%M")
        sentiment = Sentiment(
            date=dt.date().strftime("%B %d, %Y"),
            time=dt.time().strftime("%H:%M"),
            url=article.url,
            summary=article.summary,
            sentiment=trans[0]["label"],
            score=round(trans[0]["score"], 2),
        )
        all_sentiment.append(sentiment)
    return all_sentiment


def save_sentiment(all_sentiment: List[Sentiment]) -> None:
    db_file = "database.db"
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS sentiment
                      (url TEXT PRIMARY KEY, date TEXT, time TEXT, 
                       summary TEXT, sentiment TEXT, score REAL)''')
        for sentiment in all_sentiment:
            cursor.execute("INSERT OR IGNORE INTO sentiment VALUES (?,?,?,?,?,?)",
                      (sentiment.url, sentiment.date, sentiment.time, 
                       sentiment.summary, sentiment.sentiment, sentiment.score))
        conn.commit()


async def main() -> None:
    start = perf_counter()
    print("Fetching Pages...")
    all_pages = await get_pages(10)
    print("Parsing Html...")
    all_articles = parse_pages(all_pages)
    print("Calculating Sentiment...")
    all_sentiment = get_sentiment(all_articles)
    print("Saving Database...")
    save_sentiment(all_sentiment)
    end = perf_counter()
    time = end - start
    print(f"Completed in {round(time, 2)} seconds.")


if __name__ == "__main__":
    asyncio.run(main())
