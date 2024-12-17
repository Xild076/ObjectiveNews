# Objective News - The Process
By: Xild076 (Harry Yin)\
harry.d.yin.gpc@gmail.com

## 2. Introduction
Currently, misinformation is on the rise with the advent of new technology and algorithms, leading to less trust in news and less accurate news overall. I established the project **ObjectiveNews** in an attempt to combat this. As someone who does High School Policy Debate and as someone who is decently involved in politics, seeing misinformation in an increasingly polarized landscape is personally upsetting. Furthermore, studies have shown that misinformation can have real, negative effects on the human mentality[^1]. The ultimate  goal of the project is to be able to locate, group, and deliver news in the most objective way possible while staying lightweight enough as to be able to run on most devices independently. | *Currently, the locating, grouping, and delivering of the news objectively is in its beta while optimizations are being made to make it more lightweight. As of right now, the project can run on most devices, however, the time the processes take need to be improved.*
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

First, I needed to solve the issue about the clustering itself. However, I was stuck at a bottleneck here. The reason I stuck to Agglomerative Clustering was because of its `distance_threshold`. All other clustering methods only had `n_clusters`, which didn't suit my project because I needed to know how many topics there were, not give the model that information. However, that was when I came across silhoutte scores[^3]. The usage of silhoutte scores allowed me to determine the performance of a clustering with multiple different `n_clusters`, effectively giving me a way to determing the number of topics without using an arbritrary `distance_threshold`. With this, the methodology for clustering was settled.

Second, I needed to solve the issue of the long clustering times. The solution came with silhoutte score solution from above. I was testing it when I realized that I could do double level clustering. Essentially, I would cluster the text in each individual article, find their representative sentences, and cluster all the representative sentences. This would find the main idea of each cluster and see if the main idea was matched among all the other texts to form higher level groups. This would lower the amount of clustering needed to be done from something akin to $2^{1000}*1000$ to $(2^{100}*10)*100+2^{150}$ *(assuming that the clustering time is doubled per extra input, that there are 10 total texts with 100 sentences each, and a total of 15 representative sentences are retrieved per individual grouping)*. This is a massive increase in calculating efficiency, achieving my goal. With this, the methodology for improving the efficiency of the clustering was done.

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

Thus, I have the basic methodology for grouping done. *(Improvements are currently being made. Please reference Future Plans to see.)*
### 3.2 Objectifying Text

#### 3.2.1 Synonyms
### 3.3 Article Scraping and Relability

## 4. Implementation
### 4.1 Grouping
### 4.2 Objectifying
### 4.3 Article Scraping and Relability

## 5. Results
### 5.1 Grouping
### 5.2 Objectifying
### 5.3 Overall

## 6. Future Plans
1. Use this: https://medium.com/@danielafrimi/text-clustering-using-nlp-techniques-c2e6b08b6e95
    a. Maybe use attention?
2. Clean up code overall
3. Find better way to get synonyms, both in and out of context
4. Fine tune using textblob objectivity for summarization
5. Improve the rule-based objectivity
6. Improve the overall efficiency of the code, clean up unneccessary downloads
7. Clean up the UI and make it prettier
8. Fix text_fixer/remove it
9. Find better way to web scrape
10. Find better way to get keywords
## 7. Conclusion

## 8. References
[^1]: “Infodemics and Misinformation Negatively Affect People’s Health Behaviours, New Who Review Finds.” World Health Organization, World Health Organization, https://www.who.int/europe/news/item/01-09-2022-infodemics-and-misinformation-negatively-affect-people-s-health-behaviours--new-who-review-finds.
[^2]: Koch, Korbinian. “A Friendly Introduction to Text Clustering.” Medium, Towards Data Science, 27 Oct. 2022, towardsdatascience.com/a-friendly-introduction-to-text-clustering-fa996bcefd04.
[^3]: Koli, Shubham. “How to Evaluate the Performance of Clustering Algorithms Using Silhouette Coefficient.” Medium, 2 Mar. 2023, medium.com/@MrBam44/how-to-evaluate-the-performance-of-clustering-algorithms-3ba29cad8c03.