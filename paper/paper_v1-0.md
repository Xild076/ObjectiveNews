# Objective News - Finding and Determining What News to Trust
By: Xild076 (Harry Yin)\
harry.d.yin.gpc@gmail.com\
Date: 01/02/2025

### Table of Contents

## 1. Abstract

## 2. Introduction
In a growing age of information fueled by social media and greater access to technology, it is not uncommon to come across misinformation, false news, or maliciously worded facts. These often cloud the truth, harming both the individual and society. The individual may face mental distress due to online information or even physical distress after receiving incorrect information about a life-threatening topic[^1] while society is negatively affected by a build-up of polarity and poor choices. As a high school student who is somewhat informed about world politics and who does debate, I am relatively involved in determining the reliability of online information, and in doing so, I felt more that it was necessary to create some tool to prevent misinformation. Thus, I created Objective News, a project to find and present the most objective news, texts, and information possible in a growing world of misinformation. I hope that this project will help lessen the impact of misinformation in the modern age in the future.
## 3. Methodology
There are four main aspects to this project: 1. Web Scraping to retrieve information regarding a certain topic, 2. Grouping all the text into main ideas/topics of information, 3. To summarize the text into a consumable size, 4. To objectify the text as much as possible, and 5. To determine the topic's reliability.
### 3.1. Web Scraping
To begin, for web scraping, the methodology is simple. The process is split into two steps: the retrieval of links and the retrieval of the information from the websites given those links.

First, for the retrieval of links, the goal is to get the most recent news about a certain topic, and the easiest way to do so is through a Google query. The link `f"https://www.google.com/search?q={search_query}&gl=us&tbm=nws&num={amount}"` works well to efficiently do that, with a `search_query` being the query and `amount` being the number of articles displayed. Following this, requests and BeautifulSoup are used to extract the links from the page, and the links are put into the list.

Second, for the retrieval of website information, another basic method is used. Trafilatura is used to retrieve article information and if that fails, newspaper3k is used to do so. 

A rate limiter is also implemented to prevent the program from being blocked. As of right now, the rate is set to one article every three seconds.
### 3.2. Grouping
For grouping, the methodology is slightly more complicated.

First comes the textual encoding. The text is first split into sentences and preprocessed. The preprocess method is standard: removing stopwords and lemmatizing the text. This removed the excess fluff to the text, making it easier to determine the core of the text.

In textual embedding, first comes textual embedding. For textual embedding, I used a lightweight model from hugging face, the all-MiniLM-L6-v2 sentence transformer. While it isn't as detailed as models like all-mpnet-base-v2, it works well enough and the compromise is acceptable given computational constraints. 

Next comes self-attention (Authors note: I thought this was innovative at first, but it turns out there is already a paper on it, go figure)[^2]. Self-attention is a methodology used to determine the relevancy of information, and its usage would theoretically improve the clustering as the important information is singled out.

Next comes context. This is a new method used in clustering longer texts with more diverse writing. Often in text, there are short sentences or sentences that are nonsense without context. In these cases, if only one sentence is clustered, the output will sometimes be not according to what the text truly meant. Even with attention, context is necessary as attention only indicates the relevancy of the information, not the true context. Thus, context is retrieved by getting the n number of sentences around a sentence and averaging its processed embeddings with the embedding of the original sentence. That way, the algorithm has some understanding of what is happening around the sentence with extra weight given to the sentence.

Now, the embeddings are obtained and the clustering begins. The clustering algorithm is variable, it can be both `KMeans` and `AgglomerativeClustering`. The proceeding methodology uses the scoring method where clusters from 2 to max_clusters are tested and scored to determine which is the best cluster.

For scoring, three different metrics are used: silhouette, davies Bouldin, and calinski harabasz score, with each of the scores weighted and summed to form a final score. Silhouette score measures cohesion and separation, davies bouldin measures the similarity of clusters, and calinski harabasz measures between-cluster dispersion and within-cluster dispersion. Due to the variation within the ranges of the scores, min-max normalization was applied to establish some semblance of linearity within the weights.

Finally, the representative sentences of each cluster are found depending on cosine similarity, all the sentences are sorted based on their likeness to the representative sentence, and the grouping is complete.
### 3.3. Summarizing
For summarization, hugging face summarization models are used. However, the issue came down to efficiency vs performance. Due to working with computational constraints, I needed to use the most efficient model possible that still retains good performance. I tried many different models, the performances listed below.

|Model|Parameters|Performance|
|---|---|---|
|`facebook/bart-large-cnn`|406M|High accuracy w/ few mistakes
|`sshleifer/distilbart-cnn-12-6`|306M|High accuracy w/ few mistakes and formatting issues|
|`google/flan-t5-small`|77M|Medium-high accuracy w/ some acceptable mistakes|
|`google-t5/t5-small`|60.5M|Poor accuracy w/ many mistakes|

Ultimately, I chose to go with `google/flan-t5-small` which provided the best performance-to-efficiency ratio. While there are issues such as the repeat of the same phrase with different meanings, they were few enough and minor enough to be considered an acceptable compromise given the minimal number of parameters, especially when compared to `google-t5/t5-small`.

### 3.4. Objectifying

The objectifying text is a barely explored sector in NLP, so most of the methodology next presented will be just a starting point. There were two main ideas I had for objectifying text. The first idea was to fine-tune a summarization model with the Textblob subjectivity score as a metric, however, due to various circumstances, I ultimately went with the second idea: a rule-based alteration of the text to remove/replace subjective words.

First, a NLP model is used to determine the properties of the text. I decided to use Stanza's NLP model since it had a better performance than NLTK. Following that, as a general rule, any descriptive language is first checked for its objectivity and removed/replaced if it crosses a certain objectivity threshold, with objectivity being found using NLTK's wordnet.

While removing subjective words is intuitive, some words need to be replaced if they serve a key role in a sentence's structure. Synonyms of the word are found and serve to replace the subjective word with its most objective synonym. 

The location of synonyms can be done in two ways, both implemented. First is the usage of an encoder-only model to find contextual synonyms of a word. This works well for finding words that fit into the context of the sentence, however, often fail to represent the same meaning. The second method used is to use an online search to retrieve the most accurate synonyms given a part of speech with Wordnet as a backup. This method is much more reliable in achieving the same meaning of the word but sometimes misses the context of the sentence, however, considering its greater accuracy, this method is used more often.

After a word is removed, any auxiliary words, punctuations, or conjunctions that are grammatically out of place are also removed.

Any text found in quotes is not changed and all the text is joined afterwards to return more objective text. There are issues like the removal of essential, non-objective descriptive words and the incorrect determination of a word's objectivity along with grammar issues, however, the current algorithm serves as a good baseline for future endeavors.
### 3.5. Reliability
Overall, the reliability check is divided into four portions to output a numerical value to represent the reliability of a source.

The first portion is to determine the reliability of each source of the texts clustered. This is done with a massive `.csv` file with a long list of sources and their reliability score based on bias and informational accuracy, the file courtesy of nsfyn55 on GitHub [^3]. With that settled, the baseline reliability is retrieved.

Next, objectivity scores are found using Textblob. The idea is that the more objective a text is, the more likely it is to be from a more neutral and reliable source. The objectivity scores are used as a multiplier on the baseline reliabilities.

After that, general reliability is calculated, finding the overall reliability of each cluster of information. The idea for finding general reliability is that if a reliable source cites the information, then the text is probably reliable, however, if it is outnumbered severely by unreliable sources, then there is a possibility that the information was derived from primarily unreliable sources thus reducing the reliability of the information. 

To achieve this, weights are first calculated with a Gaussian weighting function with the most reliable source as a baseline. Then, the reliability is calculated by applying the weights and averaging the scores. However, to apply the idea of an outnumbered reliable source, outliers are punished by how different they are from the general score, with a weight applied depending on the severity of the penalty. Applying the penalty gives the final reliability score.

Following that, the date relevancy of the information is checked. First, more recent information is the more reliable information. Information that is old and is no longer reported on may have been disproven or may be irrelevant. Information that is too recent relative to the other news, however, may be too new to confirm its legitimacy. The most relevant and reliable information is information that has been consistently reported on for a long time.

Thus, the date relevancy score is divided into two different scores: one for coverage and one for recency. Coverage score is calculated by calculating the days covered by the given set of dates from a list of sets of dates, subtracting the smallest coverage dates from it, and averaging it over the total days all the information was covered for. 

$\text{coverage\_score} = \sqrt{\frac{\text{coverage\_days} - \text{coverage\_min}}{\text{coverage\_range}}}$ 

The square root is applied to decrease the reward given to increasing coverage. The last date score is calculated by determining the distance from the median of all the last dates and normalizing it. By applying weights to each of the scores, date relevancy is found and is used as a multiplier on the general reliability. 

Thus, reliability is calculated, with [0, 5] being considered very reliable information, (5, 15] being considered reliable, (15, 25] being considered somewhat reliable, (25, 35] being considered somewhat unreliable, and (35, inf) being considered unreliable.
## 4. Results
The results will be presented in a less numerical format for grouping and article analysis due to a lack of a proper evaluation metric for textual clustering. Objectifying will be evaluated based on the Textblob subjectivity score.
### 4.1. Grouping
Since there is no numerical evaluation for grouping, I will be subjectively evaluating the clustering based primarily on coherence and separation. 

First, grouping with no preprocessing, no attention, and no context resulted in clusters with good coherence but bad separation. There were often two or more overlapping topics, resulting in less-than-ideal clustering. 

Next, for grouping with preprocessing but no attention nor context, clusters with good coherence and passable separation were generated. There were often only two overlapping topics at max, resulting in decent clustering.

Next, for grouping with preprocessing and attention but no context, clusters with good coherence and slightly better separation were generated. There was often only one overlapping topic with attention.

Finally, with all three implemented, clusters with better coherence and similar separation were generated. There was often one overlapping topic, however, the coherence, meaning the sense of the text clustering, was much better.

Overall, while it could be better, the grouping was decent. It could be improved by fine-tuning the `max_cluster` calculation more as a smaller `max_cluster` often resulted in better separation, which is the most important improvement needed as coherence is already very good.
### 4.2. Objectifying
For objectifying, a series of subjective texts were used to find the average improvement in objectiveness. With about 300 texts, the average objectivity score was increased by 0.1389 from 0.4661 to 0.6050, being an average of a 29.80% increase in objectivity. 

An example of the objectification is: 
Before Objectification: *"That politician took horribly ineffective action that ruined the amazing nation."* (Objectivity Score: 0.3)
After Objectification: *"That politician took action that ruined the nation."* (Objectivity Score: 0.9)

Overall, the performance is good and there are few grammar mistakes. There are the occasional punctuation issues and in some cases, the text may even be scored as less objective due to the removal of key adjectives. However, in most cases, the text is further objectified.
### 4.3. Overall (Article Analysis)
Overall, article analysis was good. While there were inconsistances in some summarizations, the reliabliity scores were generally accurate and acceptable
## 5. Discussion

## 6. Future Plans

## 7. Conclusion

## 8. Sources
[^1]: https://www.who.int/europe/news/item/01-09-2022-infodemics-and-misinformation-negatively-affect-people-s-health-behaviours--new-who-review-finds
[^2]: https://arxiv.org/pdf/2201.02816
[^3]: https://gist.github.com/nsfyn55/605783ac8de36f361fb10ef187272113
https://arxiv.org/pdf/1706.03762
https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
https://medium.com/@wangdk93/multihead-attention-from-scratch-6fd6f99b9651
https://www.slingacademy.com/article/understanding-multi-head-attention-for-nlp-models-in-pytorch/
https://armanasq.github.io/nlp/self-attention/
https://arxiv.org/pdf/2211.01071
https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
https://medium.com/@srinidhikarjol/how-positional-embeddings-work-in-self-attention-ef74e99b6316
https://arxiv.org/pdf/1706.03762