# Objective News - The Process
By: Xild076 (Harry Yin)\
harry.d.yin.gpc@gmail.com

### Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Thought Process](#3-thought-process)
    1. [Grouping](#31-grouping)
    2. [Objectifying Text](#32-objectifying-text)
    3. [Article Scraping](#33-article-scraping-and-relability)
4. [Implementation](#4-implementation)
    1. [Grouping](#41-grouping)
    2. [Objectifying Text](#42-objectifying-text)
    3. [Article Scraping](#43-article-scraping-and-relability)
5. [Result](#5-results)
    1. [Grouping](#51-grouping)
    2. [Objectifying Text](#52-objectifying-text)
    3. [Article Scraping](#53-article-scraping-and-relability)
6. [Improvements since Beta](#)
7. [Future Plans](#6-future-plans)
8. [Conclusion](#7-conclusion)
9. [References](#8-references)

## 1. Abstract
In an era where misinformation and bias increasingly shape public opinion, ObjectiveNews aims to deliver a scalable, lightweight solution for generating objective news insights. This project integrates hierarchical text clustering, sentiment analysis, and web scraping to identify topics, neutralize emotive language, and assess reliability across diverse news sources. Using an innovative double-level clustering approach enhanced with silhouette scores, it efficiently groups related sentences into coherent topics. Text objectification employs rule-based techniques to minimize subjective language while preserving grammatical accuracy. Reliability is evaluated through a combination of source credibility metrics and sentiment analysis. This paper presents the methodology, challenges, and results of ObjectiveNews, showcasing its potential to combat misinformation by providing accessible, unbiased news analysis.
## 2. Introduction
Misinformation has become one of the defining challenges of the digital age. With the rise of advanced algorithms and the constant stream of information, it’s increasingly difficult to separate fact from fiction. This erosion of trust in the media and the polarization of society are not just abstract issues — they’re personal to me. As someone involved in High School Policy Debate and who closely follows politics, watching the effects of misinformation is both frustrating and alarming. Research shows how harmful misinformation can be, influencing not just public opinion but also mental health and behavior[^1].

That’s why I started ObjectiveNews — a project aimed to provide news in its most objective form. The ultimate goal is to locate, group, and deliver news objectively, while ensuring the system is lightweight enough to run on most devices without any fancy setups or extra costs. While the project is still in its beta phase, with optimizations in progress, it’s already functional on most devices. The current focus is on improving speed and refining processes, but the core concept will always remain the same: making objective news more accessible in a world that desperately needs it.
## 3. Thought Process
When I first began the project, I identified three main things I needed to do: Grouping the text, making the text objective, and gathering all that information up, summarizing it, and determining its reliability.