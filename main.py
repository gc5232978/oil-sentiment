import os
import httpx
import datetime
from rich import print
from typing import List
from dotenv import load_dotenv
from transformers import pipeline
from dataclasses import dataclass
from selectolax.parser import HTMLParser
load_dotenv()


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


def get_pages(pages: int) -> List[Page]:
    all_pages: List[Page] = []
    with httpx.Client() as client:
        for page in range(1, pages +1):
            url = f"https://oilprice.com/Latest-Energy-News/World-News/Page-{page}.html"
            resp = client.get(url).text
            page = Page(url=url, html=resp)
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


def main():
    all_pages = get_pages(5)
    all_articles = parse_pages(all_pages)
    results = get_sentiment(all_articles)
    print(results)


if __name__ == "__main__":
    main()
