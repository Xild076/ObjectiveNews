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
6. [Future Plans](#6-future-plans)
7. [Conclusion](#7-conclusion)
8. [References](#8-references)

## 1. Abstract
In an era where misinformation and bias increasingly shape public opinion, ObjectiveNews aims to deliver a scalable, lightweight solution for generating objective news insights. This project integrates hierarchical text clustering, sentiment analysis, and web scraping to identify topics, neutralize emotive language, and assess reliability across diverse news sources. Using an innovative double-level clustering approach enhanced with silhouette scores, it efficiently groups related sentences into coherent topics. Text objectification employs rule-based techniques to minimize subjective language while preserving grammatical accuracy. Reliability is evaluated through a combination of source credibility metrics and sentiment analysis. This paper presents the methodology, challenges, and results of ObjectiveNews, showcasing its potential to combat misinformation by providing accessible, unbiased news analysis.
## 2. Introduction
Misinformation has become one of the defining challenges of the digital age. With the rise of advanced algorithms and the constant stream of information, it’s increasingly difficult to separate fact from fiction. This erosion of trust in the media and the polarization of society are not just abstract issues — they’re personal to me. As someone involved in High School Policy Debate and who closely follows politics, watching the effects of misinformation is both frustrating and alarming. Research shows how harmful misinformation can be, influencing not just public opinion but also mental health and behavior[^1].

That’s why I started ObjectiveNews — a project aimed to provide news in its most objective form. The ultimate goal is to locate, group, and deliver news objectively, while ensuring the system is lightweight enough to run on most devices without any fancy setups or extra costs. While the project is still in its beta phase, with optimizations in progress, it’s already functional on most devices. The current focus is on improving speed and refining processes, but the core concept will always remain the same: making objective news more accessible in a world that desperately needs it.
## 3. Thought Process
When I first began the project, I identified three main things I needed to do: Grouping the text, making the text objective, and gathering all that information up, summarizing it, and determining its reliability.
### 3.1 Grouping
For grouping, the source I used was Korbinian Kosh's *A Friendly Introduction to Text Clustering*[^2]. The key takeaways I recieved from this articles were to: 1. embed sentences, and 2. to use hierarchical clustering. Thus, I got to work with the basic implementation of textual clustering.
```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

sentences = [...] # Put sentences here.

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

clustering_model = AgglomerativeClustering(
    distance_threshold=12.5
)

clusters = clustering_model.fit_predict(embeddings)
```
For my first attempt, I used `distance_threshold` because I didn't want to input the number of topics or clusters; I wanted the code to determine what the clusters were. For the inputs, I scraped the web for all the text I needed, put them all into one big list, and then clustered to get the main ideas.

It was good enough for a first attempt, and I tested it out. The results were lackluster. The issue was that `distance_threshold` was too sensitive and strict. Thus, if a sentence with the same topic but slightly too different of a wording was used, there would be issues. The second issue was how slow the process was. To begin with, Agglomerative Clustering was already a slow clustering method that would take exponentially longer the more inputs there were. In my case, I gave it over a thousand sentences to cluster too, which was made it ridiculously slow. Thus, I began to look for better solutions.

First, I needed to solve the issue about the clustering itself. However, I was stuck at a bottleneck here. The reason I stuck to Agglomerative Clustering was because of its `distance_threshold`. All other clustering methods only had `n_clusters`, which didn't suit my project because I needed to know how many topics there were, not give the model that information. However, that was when I came across silhouette scores[^3]. The usage of silhouette scores allowed me to determine the performance of a clustering with multiple different `n_clusters`, effectively giving me a way to determing the number of topics without using an arbritrary `distance_threshold`. With this, the methodology for clustering was settled.

Second, I needed to solve the issue of the long clustering times. The solution came with silhouette score solution from above. I was testing it when I realized that I could do double level clustering. Essentially, I would cluster the text in each individual article, find their representative sentences, and cluster all the representative sentences. This would find the main idea of each cluster and see if the main idea was matched among all the other texts to form higher level groups. This would lower the amount of clustering needed to be done from something akin to $2^{1000}*1000$ to $(2^{100}*10)*100+2^{150}$ *(assuming that the clustering time is doubled per extra input, that there are 10 total texts with 100 sentences each, and a total of 15 representative sentences are retrieved per individual grouping)*. This is a massive increase in calculating efficiency, achieving my goal. With this, the methodology for improving the efficiency of the clustering was done.

In the end, the basic framework was something like this:
```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

sentences_list = [
    [...],
    [...],
    ...
] # Put sentences here.

model = SentenceTransformer('all-MiniLM-L6-v2')

def find_representative_sentences(...):
    ...
    return representative_sentences

def cluster(sentences):
    embeddings = model.encode(sentences)
    clustering_model = AgglomerativeClustering(
        distance_threshold=12.5
    )
    clusters = clustering_model.fit_predict(embeddings)
    return clusters

representative_sentences = []
for sentences in sentences_list:
    clusters = cluster(sentences)
    representative_sentences.extend(
        find_representative_sentences(...)
    )

rep_cluster = cluster(representative_sentences)
```
Thus, I have the basic methodology for grouping done. *(Improvements are currently being made. Please reference [Future Plans](#6-future-plans) to see.)*
### 3.2 Objectifying Text
For objectifying texts, I had two primary ideas. The first idea was to fine-tune an existing model to make it produce objective texts. The plan was to have something like Textblob's textual objective scores to serve as the metric for the model, thus I began to implement. It did not go well. First, I wasn't sure how to fine-tune models. I looked for online resources, but the results were not what I needed. As a general beginner to LLMs and as someone who has only worked with the fundemental level of regression, supervised learning, and a bit of time-series models, I didn't have the knowledge necessary to implement anything. The closest I got was with a HuggingFace Blog on RLHF[^4] which gave me the theory behind it, but I was still stuck on implementation. Furthermore, while attempting another implementation of a similar fine-tuning method I found online, I realized that my device had many computational limitations. Thus, I dropped this method. *(For now. Recently, I Have delved into LLM and fine-tuning much more and believe that I will be able to implement this method soon.)*

The next idea I had was to brute force the objectifying, as in, go through every word and find which words need to be removed/altered to make the text more objective. In the english language, there are three main things that strongly convey emotion: adjectives, adverbs, and verbs. Each of them can be extremely emotive, thus those are the targets in this code. In a nutshell, if an adjective is below a certain threshold of objectivity and if the adjective is not structurally important to the text, it is removed. However, if the word is structurally important to the sentence, then a more objective synonym will be found. Similar things are done to verbs and adverbs. Given that, the most fundemental idea for objectifying text was done.
### 3.3 Article Scraping and Relability
The final biggest issues for me were webscraping and finding the reliability of information. First, webscraping was complicated. I had the limitation of being unable to use APIs due to their cost. I am attempting to keep the app entirely free and accessible, meaning anything that is priced is no good. Thus, I used a method I used in a previous project for retrieving website information: finding links using the Google search url an bs4 and using trafilatura and newspaper3k for text retrieval.

For the Google search url, I used a method I found on DEV[^5]:
```python
base_url = (
    f"https://www.google.com/search?q={search_query}"
    f"&gl=us&tbm=nws&num={amount}"
    f"&tbs=cdr:1,cd_min:{formatted_start},cd_max:{formatted_end}"
)
```
This method allowed me to find links from a certain topic in a certain range *(supposedly, the date range currently isn't working, see [Future Plans](#6-future-plans))*. The `seach_query` is derived from gathering the keywords of the title of the inputted article or keywords of the text. Thus, I have a list of links.

The next part is pretty simple, using tralifutura and then newspaper3k for backup for text retrieval and adding a rate limiter to comply with the usage policies of Google *(politely)* and to avoid being blocked *(truthfully)*.

The next issue was to find reliability. For this, my initial idea was to find reliability determined off of the sources. After looking into it, I found a great resource from nsfyn55 with a fully scrubbed media bias chart[^6]. This solved the preliminary measure for reliability, but later on, I also decided to use Textblob's subjective sentiment analysis to determine whether or not the individual text seems to be emotive or not. More emotion and subjectivity within a text usually indicates less reliability, thus I used it as a multiplier to determine the final reliability. Thus, the basic methodology for article scraping and finding reliability was established.
## 4. Implementation
However, thoughts and concepts only remain thoughts, thus I sill needed to implement of each of the three parts.
### 4.1 Grouping
First, grouping was relatively simple. Ultimately, the original [throught process](#3-thought-process) was very close to the output. However, there were two main new additions.

First was the implementation of context. An issue I noticed was that important single sentences can often be lacking in understanding of what the text truly means. Thus, to remedy this, I implemented context by gathering the sentences around the single sentence to form a paragraph with three sentences, combining the two embeddings with weights, and then inputting that into the cluster. Whiel I'm only making assumptions that this works, and I'm not sure why or how it works, from testing it, it seems to produce good results.

Second was the usage of more scoring metrics. At the beginning, I only used silhouette scoring, which measure the closeness of points in the same cluster and the distance of points of different clusters. However, I found out about two other cluster metrics: davies-bouldin, which measures the average similarity ratio of clusters, and calinski-harabasz, which measures ratio of the sum of between-cluster dispersion to within-cluster dispersion. Each of these metrics are important in their own way as text clustering can also be evaluated on different clusters. Since my goals are to create distinct but general clusters, silhouette and davies-bouldin scores ended up being the most important for evaluation, however, I still made the weights changable. Due to each score having a different range, I needed to play around with the score to make the final caculation work, meaning that any change in the weights could cause extremely volitile reactions due to the inverse calculations done. I got lucky with `{'sil': 0.4, 'db': 0.55, 'ch': 0.05}` being a good series of weights, however, work still needs to be done to find a better calculation for scoring *(Please reference [Future Plans](#6-future-plans))*
### 4.2 Objectifying
Next, objectifying was silghtly more complicated due to something we all love/hate/feel indifferent to: Grammar Rules (and NLP).

To begin, the implementation of the removal of adjectives was simple. For NLP, I primarily used Stanford Stanza since it does part-of-speech analysis the best, which is the most important for my purposes. Next, I needed to split the text into segments based on quotes. Even if I was trying to make the text more objective, quotes needed to be preserved, thus the text was split using `re`.

Now, with the segments, I began to iterate through all the words and analyse each one of the words, done through three metrics: the part of speech, the objectivity of the word, and the position of the word. The adjectives, adverbs, and verbs were handled as mentioned in the [thought process](#3-thought-process): they were either removed, left along, or replaced with a synonym depending on how objective they were and their location in the text.

The issue came after the adjectives were removed. After the words were removed, there would be hanging punctuation, determiners, and conjunctions. These needed to be removed, which resulted in a long and arduous journey figuring out the grammatical rules that would necessitate the removal of any of the hanging texts. Ultimately, the code came together with this:
```python
add = 0 # add is 0, remove is 1, synonym is 2
word_text = word.text
word_pos = word.upos
word_objectivity = calc_objectivity_word(word_text, get_pos_wn(word_pos))
if word_pos == "ADJ":
    if word_objectivity < objectivity_threshold: # If it is above the threshold, add normally, no need to change
        if sentence.words[i-1].upos == "AUX": # If preceding is AUX (is, was, etc. -> They were ____), add synonym
            add = 2
        else:
            n = i + 1
            while n < len(sentence.words):
                if sentence.words[n].upos in {"NOUN", "PROPN"}: # If subsequent word is Noun/Proper Noun, remove it
                    add = 1
                    break
                if sentence.words[n].upos not in {"CCONJ", "CONJ", "PUNCT", "ADJ"}: # If subsequent word is not connecting (conj, punct) or adjective, remove it
                    break
                n += 1
elif word_pos == "VERB":
    if word_objectivity < objectivity_threshold:# If it is above the threshold, add normally, no need to change
        add = 2
elif word_pos == "ADV":
    if 1 <= i <= len(sentence.words) - 2:
        prev_token = sentence.words[i-1].upos
        next_token = sentence.words[i+1].upos
        if prev_token != "AUX":
            if next_token == "ADJ":
                if calc_objectivity_word(sentence.words[i+1].text, get_pos_wn(next_token)) < objectivity_threshold:
                    add = 1
            elif word_objectivity < objectivity_threshold:
                add = 1
elif word_pos in {"CONJ", "CCONJ"}:
    if 1 <= i <= len(sentence.words) - 2:
        prev_token = sentence.words[i-1].upos
        next_token = sentence.words[i+1].upos
        if prev_token in {"ADJ", "PUNCT"} and next_token in {"ADJ"}: # If it is surrounded by lists of adjectives
            if (calc_objectivity_word(sentence.words[i-1].text, get_pos_wn(prev_token)) < objectivity_threshold or 
                calc_objectivity_word(sentence.words[i+1].text, get_pos_wn(prev_token)) < objectivity_threshold): # Either or: Conj is less important in grammatical fluency
                add = 1
elif word_pos in {"PUNCT"}:
    if 1 <= i <= len(sentence.words) - 2:
        prev_token = sentence.words[i-1].upos
        next_token = sentence.words[i+1].upos
        if prev_token in {"ADJ", "PUNCT"} and next_token in {"ADJ"}: # If it is surrounded by lists of adjectives
            if (calc_objectivity_word(sentence.words[i-1].text, get_pos_wn(prev_token)) < objectivity_threshold and 
                calc_objectivity_word(sentence.words[i+1].text, get_pos_wn(prev_token)) < objectivity_threshold): # And: Conj is less important in grammatical fluency
                add = 1
        if prev_token in {"ADV"}: # If an adverb is before it
            if calc_objectivity_word(sentence.words[i-1].text, get_pos_wn(prev_token)) < objectivity_threshold:
                add = 1

if add == 0:
    processed_sentence.append(word_text)
elif add == 2:
    processed_sentence.append(get_objective_synonym(word_text, sentence.text))
```
The issue is, it is still incomplete. Despite running many test cases, I still continously find new errors. Furthermore, the text post-processing of combining the text back into a legible sentence still needs improvements. There are often random spaces or not enough spaces due to the weird format of characters. I plan to rework this in the future too *(See [Future Plans](#6-future-plans))*.
#### 4.2.1 Word Objectivity
For word objectivity, I was at first going to use Textblob's subjective score. The issue was that the subjective score didn't work for individual words. The ultimate implementation of word objectivity came with nltk's wordnet. It was relatively simple code to implement, although it is inconsistent and doesn't work for all words. However, it is good enough as of right now.
```python
def calc_objectivity_word(word, pos=None):
    obj_scores = []
    synsets = wn.synsets(word, pos=pos)
    for syn in synsets:
        try:
            swn_syn = swn.senti_synset(syn.name())
            obj_scores.append(swn_syn.obj_score())
        except Exception as e:
            pass
    if obj_scores:
        obj_swn = sum(obj_scores)/len(obj_scores)
    else:
        obj_swn = 0.5
    return obj_swn
```
#### 4.2.2 Synonyms
Finding appropriate synonyms for a word in context was slightly harder. First, I used wordnet to find synonyms, however, the synonyms, being honest, were not that good. They often failed to find objective version, having similarily exaggerated words as the original sentence. Second, I used a encoder-decoder model, namely roberta-large, to try to find synonyms. However, once again, the synonyms found were only in the context of the sentence, not with the same meaning. Finally, I used gpt2 to find synonyms, and it worked decently. The issue was that gpt2 is massive, meaning there are computational errors. Ultimately, I used a combination of all three to find synonyns, however, I still plan to focuse on encoder-decoder models for the solution as I believe if I give it a proper 1 or 2-shot inference, it might work better *(See [Future Plans](#6-future-plans))*.
### 4.3 Article Scraping and Relability
There isn't much to say for article scraping. The implementation was the exact same as the [thought process](#33-article-scraping-and-relability), however, there were some small alterations to the reliability calculation. The issue was the reliability calculation. At first, I thought that just averaging out the reliability scores of all the sources would work, however, that is not the case. The problem is, many sources sometimes report about the same thing, some reliable, some not. Most people would consider that information reliable, however, the code, finding the average, would not. Thus, the ultimate equation formed was $\frac{\min(R) * 5 + \sum R}{\text{len}(R) + 5}$ where R is a list of the reliability scores. However, I still think it could be improved *(See [Future Plans](#6-future-plans))*.
## 5. Results
After implementing everything, the finale comes: testing and results.
### 5.1 Grouping
Grouping works pretty well. In a vacuum, it is the fastest of all the processes. In terms of pure textual clustering, it does a good job in dividing text into many ideas. However, there are still issues with grouping overlaps, which is something that needs to be improved on. The usage of context is still questionable and any changes to the metric weights can result in the entire code being broken. This is an issue that needs to be fixed, however, using the correct weights, the results are pretty consistant.
### 5.2 Objectifying
Objectifying generally does its job. There is a consistent 5-10% increase in Textblob objectivity, which is a pretty big improvement. However, it takes too long and is also sometimes grammatically inconsistent with broken outputs. First, for the time length, the issue lies in the fact that big models are used to find synonyms and that each and every word is iterated through. This takes too long and improvements can be made, along with the rejoining of the text.
### 5.3 Overall
Overall, the outputs are good but leave a lot to be desired. The clustering is inconsistent, the summarizations are , and the processes are still not resource/time efficient enough. As of right now, a single analysis can take upwards of 2 minutes, which is too slow. The majority of the issue comes with the objectifying process. The issue is that between the two large models used, one for summarization and one for finding synonyms. Other than that, there are just overall quality changes and maybe revampts that can be made to improve the methodology of the clustering. However, as a beta, it is good enough.
## 6. Future Plans
This project, as seen with the results, still has many things that need to be improved on.
1. For grouping, the embedding method could definitely be improved. The current method still lacks vectorization, reference: [Text Clustering Using NLP](https://medium.com/@danielafrimi/text-clustering-using-nlp-techniques-c2e6b08b6e95), and after doing research on LLMs, attention could be a possible addition.
2. For objectifying, clean up the text post-processing. Right now, there are inconsistancies with quote placement and missing/extra spaces. I plan to move to a purely algorithmic method for fixing inconsistancies instead of using `re` more, as `re` fails to consider many of same-character differences in formatting.
3. For synonyms, use a more lightweight way to implement it and maybe use methods like 1 or 2-shot inferences or some other methods with a smaller model.
4. Also for objectifying and summarizations, fine-tune a lightweight model instead of using rule-based objectivity.
5. For article retrieval, fix issues with the scraper time-frame and the keyword selection.
6. Create a better way to calculate relability from current methods
    a. Author credentials
    b. Domain Authority and SEO Metrics
    c. Technical Indicators of Fake News Sites
    d. ML-based fake news detection
7. Temporal analysis to find the recency of the news?
6. Fix the entire `text_fixer.py` file since 
7. Just clean up the code overall and add proper documentation.
8. Improve the Streamlit UI with cooler/cleaner visuals
9. Multilingual support
## 7. Conclusion
Overall, the project is in its early stages. There are many issues, however, I believe that there is a solid foundation in place. As I learn more, I will continue to update and improve this project. I hope one day, this tool will serve as a method to help people avoid falling into the trap of misinformation.
## 8. References
[^1]: “Infodemics and Misinformation Negatively Affect People’s Health Behaviours, New Who Review Finds.” World Health Organization, World Health Organization, who.int/europe/news/item/01-09-2022-infodemics-and-misinformation-negatively-affect-people-s-health-behaviours--new-who-review-finds.
[^2]: Koch, Korbinian. “A Friendly Introduction to Text Clustering.” Medium, Towards Data Science, 27 Oct. 2022, towardsdatascience.com/a-friendly-introduction-to-text-clustering-fa996bcefd04.
[^3]: Koli, Shubham. “How to Evaluate the Performance of Clustering Algorithms Using Silhouette Coefficient.” Medium, Medium, 2 Mar. 2023, medium.com/@MrBam44/how-to-evaluate-the-performance-of-clustering-algorithms-3ba29cad8c03.
[^4]: Lambert, Nathan et al, “Illustrating Reinforcement Learning From Human Feedback (RLHF).” HuggingFace, HuggingFace, 9 Dec. 2022, huggingface.co/blog/rlhf.
[^5]: Serpdog. “Web Scraping Google News Using Python.” DEV Community, DEV, 18 Oct. 2024, dev.to/serpdogapi/web-scraping-google-news-using-python-5f86.
[^6]: “A Fully Scrubbed CSV of All of Media Bias Fact Check’s Primary Categories(Note on Bias Negative(-) Connotes Liberal Bias, Positive(+) Connotes Conservative Bias).” Gist, gist.github.com/nsfyn55/605783ac8de36f361fb10ef187272113.