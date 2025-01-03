# Objective News - Finding and Determining What News to Trust
By: Xild076 (Harry Yin)\
harry.d.yin.gpc@gmail.com\
Date: 01/02/2025

### Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Methodology](#3-methodology)
    1. [Web Scraping](#31-web-scraping)
    2. [Grouping](#32-grouping)
    3. [Summarizing](#33-summarizing)
    4. [Objectifying](#34-objectifying)
    5. [Reliability](#35-reliability)
5. [Results](#5-results)
6. [Discussion](#6-discussion)
7. [Future Plans](#7-future-plans)
8. [Conclusion](#8-conclusion)
9. [Sources](#9-sources)

## 1. Abstract

## 2. Introduction
In a growing age of information fueled by social media and greater access to technology, it is not uncommon to come across misinformation, false news, or maliciously worded facts. These often cloud the truth, harming both the individual and society. The individual may face mental distress due to online information or even physical distress after recieving incorrect information about a life-threatening topic[^1] while society is negatively affected from a build-up of polarity and poor choices. As a high school student who is somewhat informed in world politics and who does debate, I am relatively involved in determining the reliability of online information, and in doing so, I felt more so that it was necessary to create some tool to prevent misinformation. Thus, I created Objective News, a project with the goal of finding and presenting the most objective news, texts, and information possible in a growing world of misinformation. I hope that this project will help lessen the impact of misinformation in the modern age in the future.
## 3. Methodology
There are four main aspects to this project: 1. Web Scraping to retrieve the information regarding a certain topic, 2. Grouping all the text into main ideas / topics of information, 3. To summarize the text into a consumable size, 4. To objectify the text as much as possible, and 5. To determine the topic's reliability.
### 3.1. Web Scraping
To begin, for web scraping, the methodology is simple. The process is split into two steps: the retrieval of links and the retrieval of the information from the websites given those links.

First, for the retrieval of links, the goal is to get the most recent news about a certain topic, and the easiest way to do so was through a google query. The link `f"https://www.google.com/search?q={search_query}&gl=us&tbm=nws&num={amount}"` works well to efficiently do that, with a `search_query` being the query and `amount` being the number of articles displayed. Following this, requests and BeautifulSoup is used to extract the links from the page and the links are put into the list.

Second, for the retrieval of website information, another basic method is used. Trafilatura is used to retrieve article information and if that fails, newspaper3k is used to do so. 

A rate limiter is also implemented to prevent the program from being blocked. As of right now, the rate is set to one article every three seconds.
### 3.2. Grouping
For grouping, the methodology is slightly more complicated.

First comes the textual encoding. The text is first split into sentences and preprocessed. The preprocess method is standard: removing stopwards and lemmatizing the text. This removed the excess fluff to the text, making it easier to determing the core of the text.

Next comes textual embedding. For textual embedding, I used a lightweight model from huggingface, the all-MiniLM-L6-v2 sentence transformer. While it isn't as detailed as models like all-mpnet-base-v2, it works well enough and the comprimise is acceptable given computational constraints. 

Next comes self-attention (Authors note: I thought this was innovative at first, but turns out there is already a paper on it, go figure)[^2]. Self-attention is a methodology used to determine the relevancy of information, and its usage would theoretically improve the clustering as the important information is singled out.

Next comes context. This is new method used in clustering longer texts with more diverse writing. Often in text, there are short sentences or sentences that are nonsense without context. In these cases, if only one sentence is clustered, the output will sometimes be not according to what the text truly meant. Even with attention, context is necessary as attention only indicates the relevancy of the information, not the true context. Thus, context is retrieved by getting the n number of sentences around a sentence and averaging its processed embeddings with the embedding of the original sentence. That way, the algorithm has some understanding of what is happening around the sentence with extra weight given to the sentence.
### 3.3. Summarizing
For summarization, huggingface summarization models are used. However, the issue came down to efficiency vs performance.
### 3.4. Objectifying

### 3.5. Reliability

- Most reliable source sets baseline, the lower the more the score leans low
- Less weight to unreliable sources, the higher the score the less it contributes
- Reality check, if too many unreliable sources, it adds a penalty
- Balance between reliable and unreliable, strong reliable source pulls the final score towards reliability, multiple unreliable sources can still drag down the reliability if they outweigh the good ones
- It balances optimism from the star player (reliable sources) with realism about the team’s overall strength (unreliable sources)

## 5. Results

### 5.1. Grouping

### 5.2. Objectifying

### 5.3. Summarizing

### 5.4. Overall (Article Analysis)

## 6. Discussion

## 7. Future Plans

## 8. Conclusion

## 6. Discussion

## 7. Future Plans

## 8. Conclusion

## 9. Sources
[^1] https://www.who.int/europe/news/item/01-09-2022-infodemics-and-misinformation-negatively-affect-people-s-health-behaviours--new-who-review-finds
[^2] https://arxiv.org/pdf/2201.02816
https://arxiv.org/pdf/1706.03762
https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
https://medium.com/@wangdk93/multihead-attention-from-scratch-6fd6f99b9651
https://www.slingacademy.com/article/understanding-multi-head-attention-for-nlp-models-in-pytorch/
https://armanasq.github.io/nlp/self-attention/
https://arxiv.org/pdf/2211.01071
https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
https://medium.com/@srinidhikarjol/how-positional-embeddings-work-in-self-attention-ef74e99b6316
https://arxiv.org/pdf/1706.03762