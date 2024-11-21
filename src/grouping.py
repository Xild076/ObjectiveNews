from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from util import preprocess_text, get_keywords, find_bias_rating
import numpy as np
from nltk.tokenize import sent_tokenize
from scaper import FetchArticle
from datetime import datetime, timedelta
from typing import Literal
import validators

class GroupText:
    model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

    @staticmethod
    def _find_representative_sentences(sentences, embeddings, labels):
        cluster_embeddings = {l:[] for l in labels}
        cluster_sentences = {l:[] for l in labels}
        for sentence, embedding, label in zip(sentences, embeddings, labels):
            cluster_embeddings[label].append(embedding)
            cluster_sentences[label].append(sentence)
        representative_sentences = {}
        for label in cluster_embeddings:
            cluster_embeds = np.vstack(cluster_embeddings[label])
            centroid = np.mean(cluster_embeds, axis=0)
            distances = np.linalg.norm(cluster_embeds - centroid, axis=1)
            min_idx = np.argmin(distances)
            representative_sentences[label] = cluster_sentences[label][min_idx]
        return representative_sentences

    @staticmethod
    def _group_text_individual(text):
        sentences = [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]
        if not sentences:
            return {}, {}
        processed_sentences = [preprocess_text(sentence) for sentence in sentences]
        embeddings = GroupText.model.encode(processed_sentences)
        num_sentences = len(sentences)
        if num_sentences < 2:
            labels = np.zeros(num_sentences, dtype=int)
        else:
            max_clusters = min(4, num_sentences - 1)
            best_score = -1
            best_labels = None
            for n_clusters in range(3, max_clusters + 1):
                clustering_algc = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clustering_algc.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
            labels = best_labels
        cluster_sentences = {l:[] for l in labels}
        for sentence, label in zip(sentences, labels):
            cluster_sentences[label].append(sentence)
        representative_sentences = GroupText._find_representative_sentences(sentences, embeddings, labels)
        return cluster_sentences, representative_sentences

    @staticmethod
    def group_text(texts):
        all_representative_sentences = []
        for text in texts:
            _, rep_sent = GroupText._group_text_individual(text)
            all_representative_sentences.extend(rep_sent.values())
        if not all_representative_sentences:
            return {}, {}
        processed_sentences = [preprocess_text(sentence) for sentence in all_representative_sentences]
        embeddings = GroupText.model.encode(processed_sentences)
        num_sentences = len(all_representative_sentences)
        if num_sentences < 2:
            labels = np.zeros(num_sentences, dtype=int)
        else:
            max_clusters = min(4, num_sentences - 1)
            best_score = -1
            best_labels = None
            for n_clusters in range(3, max_clusters + 1):
                clustering_algc = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clustering_algc.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
            labels = best_labels
        cluster_sentences = {l:[] for l in labels}
        for sentence, label in zip(all_representative_sentences, labels):
            cluster_sentences[label].append(sentence)
        representative_sentences = GroupText._find_representative_sentences(all_representative_sentences, embeddings, labels)
        return cluster_sentences, representative_sentences

    @staticmethod
    def article_analyse(link, type:Literal['news', 'data']='news'):
        if not validators.url(link):
            raise TypeError(f"Link {link} is not valid")
        article = FetchArticle.extract_article_details(link)
        text = article['text']

        keywords = get_keywords(text)
        if type == 'news':
            date_range = 2
        elif type == 'data':
            date_range = 30
        start_date = (datetime.strptime(article['date'], '%Y-%m-%d') - timedelta(days=date_range)).strftime('%Y-%m-%d')
        end_date = (datetime.strptime(article['date'], '%Y-%m-%d') + timedelta(days=date_range)).strftime('%Y-%m-%d')
        links = FetchArticle.retrieve_links(keywords, start_date, end_date, 10)

        articles = FetchArticle.extract_many_article_details(links)
        

GroupText.article_analyse('https://www.cbsnews.com/news/matt-gaetz-depositions-leak-investigations/')