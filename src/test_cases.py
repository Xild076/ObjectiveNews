import numpy as np
import nltk
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sentence_transformers import SentenceTransformer
from typing import Union, Type, List, Dict, Any
from nltk.stem import WordNetLemmatizer
from colorama import Fore, Style
from grouping import cluster_text
from article_analysis import cluster_articles, organize_clusters, provide_metrics
from summarizer import summarize_text
from objectify_text import objectify_text


mangione_case_details = [
    "They have been left stunned by the 26-year-old’s arrest for the murder of UnitedHealthcare’s chief executive Brian Thompson, who was fatally shot last week in New York City. Mr Mangione will plead not guilty, his lawyer says. According to a law enforcement bulletin seen by US media, Mr Mangione was allegedly motivated by resentment at what he called 'parasitic' health insurance companies.",
    
    "Mr Mangione will plead not guilty, his lawyer says. According to a law enforcement bulletin seen by US media, Mr Mangione was allegedly motivated by resentment at what he called 'parasitic' health insurance companies. He had spent time in a surfing community in Hawaii, but left owing to debilitating back pain, say those who remember him.",
    
    "Mr Martin - who eventually lost contact with Mr Mangione - said that he believed his former friend 'would have never conceived of hurting someone else.' 'There’s no making sense of it,' he added. A person matching his name and photo had an account on Goodreads, a user-generated book review site, where he read two books about back pain in 2022, one of them called Crooked: Outwitting the Back Pain Industry.",
    
    "Mangione appears to have been driven by anger against the industry and 'corporate greed' as a whole, states a New York Police Department intelligence report that also warns online rhetoric could 'signal an elevated threat facing executives in the near-term.' 'That horrific attack occurred on our streets,' Hochul said Thursday. 'And the people of our city deserve to have that sense of calm that this perpetrator has been caught, and he will be never seeing the late of day again if there is justice.' 'You cannot assassinate an individual on the streets of New York – not now, not ever.'",
    
    "Until he was found Monday in Altoona, Pennsylvania, Mangione had gone quiet from at least a few people who appear to have been close with him. The scion of a wealthy Baltimore family who was a high school valedictorian and an Ivy League graduate, he had maintained an active social media presence for years, posting smiling photos from his travels, sharing his weightlifting routine and discussing health challenges he faced.",
    
    "An image posted to a social media account linked to Mangione showed what appeared to be an X-ray of a metal rod and multiple screws inserted into someone’s lower spine. Martin stopped hearing from Mangione six months to a year ago. An X account linked to Mangione includes recent posts about the negative impact of smartphones on children; healthy eating and exercise habits; psychological theories; and a quote from Indian philosopher Jiddu Krishnamurti about the dangers of becoming 'well-adjusted to a profoundly sick society.'",
    
    "Police report a darker turn. Mangione likely was motivated by his anger at what he called 'parasitic' health insurance companies and a disdain for corporate greed, according to a law enforcement bulletin obtained by AP.",
    
    "They donated to various causes, including Catholic organizations, colleges and the arts. One of Luigi Mangione’s cousins is Republican Maryland state legislator Nino Mangione, a spokesman for the lawmaker’s office confirmed. 'Our family is shocked and devastated by Luigi’s arrest,' Mangione’s family said in a statement posted on social media by Nino Mangione.",
    
    "The Mangione family also purchased Hayfields Country Club north of Baltimore in 1986. On Monday, Baltimore County police officers blocked off an entrance to the property, which public records link to Luigi Mangione’s parents. Reporters and photographers gathered outside the entrance.",
    
    "6:44 a.m.: Thompson is killed. He is shot from behind and is then shot more times. Three pieces of ammunition recovered had 'deny,' 'delay' and 'depose' written on them in marker, one on each, Kenny said.",
    
    "Tuesday, Dec. 10, 1:40 p.m.: Mangione struggles with guards as they try to lead him into Blair County, Pennsylvania, court. 'It’s completely out of touch and an insult to the intelligence of the American people and their lived experience,' he yells as he tussles with three guards who push him into the courthouse. He is denied bail and vows to fight extradition to New York.",
    
    "Police also told CBS News that fingerprints on a phone found at the scene are a match to Mangione’s. Investigators also matched a 'ghost gun' that police found with Mangione in Pennsylvania to three 9 mm shell casings from the shooting in New York, according to the NYPD. The fingerprint match was first reported by CNN.",
    
    "Fox News’ David Hammelburg contributed to this report. Fox News Digital has obtained the mugshot of UnitedHealthcare CEO murder suspect Luigi Mangione, as well as a New York arrest warrant that says Mangione had 'written admissions about the crime' when he was arrested Monday in Pennsylvania.",
    
    "The website reports that for a time, Mangione lived in the 'Surfbreak' co-working and co-living space located inside the Century Square condo building in Honolulu. 'Surfbreak HNL is the first co-living and co-working penthouse for remote workers in Hawaii,' reads a description of the shared penthouse on its website. 'Our 40th floor floor-to-ceiling glass space offers panoramic views… from city life to surf sesh and mountain heights.'",
    
    "Pennsylvania State Police released new photos that show for the first time murder suspect Luigi Mangione pictured inside the McDonald’s where he was apprehended in Altoona, Penn. Mangione is seen wearing a medical face mask and eating what looks like a McDonald’s hashbrown in the corner of the fast food chain."
]

text = " ".join(mangione_case_details).replace("\'", "'")
num_sentences = len(mangione_case_details)
summary_length_min = round(len(text) / (num_sentences + 3) / 3)
summary_length_max = round(len(text) / (num_sentences - 3) / 3)
print(summary_length_min, summary_length_max)
print(summarize_text(text, min_length=summary_length_min, max_length=summary_length_max))

"""url = 'https://www.bbc.com/news/articles/cp9nxee2r0do'
cluster_article = cluster_articles(url, link_num=5)
print(cluster_article)
metrics = provide_metrics(cluster_article)
print(metrics)
organized = organize_clusters(metrics)
print(organized)

file_path = "output.txt"
with open(file_path, "w") as file:
    file.write("Cluster Article:\n")
    file.write(f"{cluster_article}\n\n")
    file.write("Metrics:\n")
    file.write(f"{metrics}\n\n")
    file.write("Organized Clusters:\n")
    file.write(f"{organized}\n")

print(f"Data saved to {file_path}")"""