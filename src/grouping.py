from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sentence_splitter import SentenceSplitter
from transformers import pipeline, AutoTokenizer
import validators
from scaper import FetchArticle
from datetime import datetime, timedelta
from util import get_keywords


class GroupText:
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    summarizer = pipeline('summarization')
    summarizer_model_name = summarizer.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)

    
    @staticmethod
    def group_text(texts:list, dist_threshold=50, _visualize=False):
        all_sentences = []
        sentence_info = []
        unique_sources = set()
        for text in texts:
            sentences = SentenceSplitter(language='en').split(text['text'])
            all_sentences.extend(sentences)
            for sentence in sentences:
                sentence_info.append({sentence: text['source']})
                unique_sources.add(text['source'])
        
        embeddings = GroupText.model.encode(all_sentences)
        clustering = AgglomerativeClustering(distance_threshold=dist_threshold, n_clusters=None)
        clustering.fit(embeddings)
        labels = clustering.labels_

        cluster_sentences = {label: [] for label in set(labels)}
        cluster_embeddings = {label: [] for label in set(labels)}

        for sentence, label, embedding in zip(sentence_info, labels, embeddings):
            cluster_sentences[label].append(sentence)
            cluster_embeddings[label].append(embedding)

        representative_sentences = {}
        for label in cluster_sentences:
            sentences = cluster_sentences[label]
            sentences_texts = [list(sentence.keys())[0] for sentence in sentences]
            cluster_text = ' '.join(sentences_texts)
            max_input_length = 1024
            tokens = GroupText.tokenizer.encode(
                cluster_text, truncation=True, max_length=max_input_length)
            cluster_text = GroupText.tokenizer.decode(
                tokens, skip_special_tokens=True)

            summary = GroupText.summarizer(
                cluster_text, max_length=50, min_length=5, do_sample=False)

            representative_sentences[label] = summary[0]['summary_text']
        
        if _visualize:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=labels, cmap='viridis', s=100)
            ax.set_title('3D Visualization of Sentence Clusters')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            plt.colorbar(scatter, ticks=range(len(set(labels))), label='Cluster Label')

            for i, txt in enumerate(all_sentences):
                ax.text(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2], txt, size=10, zorder=1, color='k')

            plt.show()

        
        return cluster_sentences, representative_sentences
    
    def article_analyse(link, type='news'):
        if not validators.url(link):
            raise TypeError(f"Link {link} is not valid")
        article = FetchArticle.extract_article_details(link)
        text = article['text']

        keywords = get_keywords(text)

        print("Keywords", keywords)

        if type == 'news':
            date_range = 5
        else:
            date_range = 30
        start_date = (datetime.strptime(article['date'], '%Y-%m-%d') - timedelta(days=date_range)).strftime('%Y-%m-%d')
        end_date = (datetime.strptime(article['date'], '%Y-%m-%d') + timedelta(days=date_range)).strftime('%Y-%m-%d')
        links = FetchArticle.retrieve_links(keywords, start_date, end_date)

        print(links)

        articles = FetchArticle.extract_many_article_details(links)

        cluster_s, rep_s = GroupText.group_text(articles, 50, True)

        print(rep_s)


GroupText.article_analyse("https://www.cnn.com/politics/live-news/election-trump-harris-11-06-24/index.html")