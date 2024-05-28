# oil-sentiment

Gets the current sentiment of the crude oil market using transformers.

## Description

This project scrapes news summaries from:
```
oilprice.com
```
Performs sentiment analysis using:
```
mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
```
Model information:
```
https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
```

## Getting Started

### Dependencies

* httpx
* selectolax
* rich
* transformers

### Installing

First make sure you have pytorch installed.
```
torch --version
```
then run
```
git clone https://github.com/gc5232978/oil-sentiment.git
cd oil-sentiment
poetry install
```

### Executing program

```
python main.py
```