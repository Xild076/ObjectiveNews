import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings

# Suppress the sklearn deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

print("Loading models...")

# Initialize lemmatizer and model
lemmatizer = WordNetLemmatizer()
model = SentenceTransformer('all-mpnet-base-v2')

print("Models loaded successfully.")

def preprocess_text(text: str, language='english') -> str:
    """
    Preprocess text by tokenizing, lemmatizing, and removing stop words.
    
    Args:
        text: Input text to preprocess
        language: Language for stop words (default: 'english')
    
    Returns:
        str: Preprocessed text
    """
    try:
        stop_words = set(stopwords.words(language))
    except LookupError:
        print(f"Warning: NLTK stopwords for '{language}' not found. Using empty set.")
        stop_words = set()
    
    # Clean and tokenize text
    text = text.lower().replace("\n", " ").replace("\t", " ").replace("\r", " ")
    tokens = word_tokenize(text)
    
    # Lemmatize and filter stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word.isalpha() and word not in stop_words]
    
    return ' '.join(tokens)

def encode_text(sentences, weights, context, context_len, preprocess):
    """
    Encode sentences into embeddings with optional context weighting.
    
    Args:
        sentences: List of sentences to encode
        weights: Dict with 'single' and 'context' weights
        context: Whether to use context embeddings
        context_len: Number of sentences before/after to include in context
        preprocess: Whether to preprocess text
    
    Returns:
        np.ndarray: 2D array of embeddings
    """
    if preprocess:
        sentences = [preprocess_text(s) for s in sentences]
    
    embeddings = model.encode(sentences, show_progress_bar=True)
    
    if not context:
        return embeddings
    
    final_embeddings = []
    for i, emb in enumerate(embeddings):
        # Get context embeddings
        start_index = max(0, i - context_len)
        end_index = min(len(embeddings), i + context_len + 1)
        context_embeddings = embeddings[start_index:end_index]
        context_mean = context_embeddings.mean(axis=0)
        
        # Combine single embedding with context
        weighted_emb = (emb * weights["single"]) + (context_mean * weights["context"])
        final_embeddings.append(weighted_emb)
    
    return np.array(final_embeddings)

def find_representative_sentence(X: np.ndarray, labels: np.ndarray, cluster_label: int) -> int:
    cluster_indices = np.where(labels == cluster_label)[0]
    cluster_points = X[cluster_indices]
    if len(cluster_points) == 0:
        raise ValueError(f"No points found for cluster {cluster_label}.")
    centroid = cluster_points.mean(axis=0).reshape(1, -1)
    similarities = cosine_similarity(cluster_points, centroid).flatten()
    rep_relative_idx = np.argmax(similarities)
    rep_idx = cluster_indices[rep_relative_idx]
    return rep_idx

def cluster_sentences(sentences, weights, context=False, context_len=0, preprocess=True):
    """
    Cluster sentences using HDBSCAN algorithm.
    
    Args:
        sentences: List of sentences to cluster
        weights: Dict with 'single' and 'context' weights
        context: Whether to use context embeddings
        context_len: Number of sentences before/after to include in context
        preprocess: Whether to preprocess text
    
    Returns:
        tuple: (labels, representative_indices)
    """
    print("Clustering sentences...")
    if len(sentences) == 0:
        return np.array([]), []

    embeddings = encode_text(sentences, weights, context, context_len, preprocess)
    
    # Normalize embeddings for better cosine similarity behavior with euclidean distance
    from sklearn.preprocessing import normalize
    embeddings_normalized = normalize(embeddings, norm='l2')
    
    # Use euclidean distance on normalized embeddings (equivalent to cosine similarity)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,  # Small clusters for demonstration
        min_samples=1,       # Allow single points to form clusters
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(embeddings_normalized)

    unique_labels = np.unique(labels)
    representative_indices = []
    
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        rep_idx = find_representative_sentence(embeddings_normalized, labels, label)
        representative_indices.append(rep_idx)
    
    print(f"Clustering completed. Found {len(representative_indices)} clusters.")
    return labels, representative_indices

def visualize_clusters(embeddings, labels, title="HDBSCAN Clustering of Sentence Embeddings"):
    """
    Visualize clusters using PCA dimensionality reduction.
    
    Args:
        embeddings: 2D array of embeddings
        labels: Cluster labels
        title: Plot title
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:  # Noise points
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                       c='black', marker='x', s=50, alpha=0.6, label='Noise')
        else:
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                       c=[colors[i]], s=60, alpha=0.8, label=f'Cluster {label}')
    
    plt.title(title)
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate sentence clustering.
    """
    # Sample text with more similar sentences for better clustering demonstration
    text = """
By  SEAN MURPHY, NADIA LATHAN and JOHN SEEWER
Updated 2:59 PM PDT, July 9, 2025
Share
HUNT, Texas (AP) — In the frantic hours after a wall of water engulfed camps and homes in Texas, a police officer who was trapped himself spotted dozens of people stranded on roofs and waded out to bring them to safety, a fellow officer said Wednesday.

Another off-duty officer tied a garden hose around his waist so he could reach two people clinging to a tree above swirling floodwaters, Kerrville officer Jonathan Lamb said, describing another harrowing rescue.

Eric Herr, a volunteer with Search and Support San Antonio, does search and rescue work on the banks of the Guadalupe River in Ingram, Texas, Tuesday, July 8, 2025, after the Fourth of July flood. (Jay Janner/Austin American-Statesman via AP)
Eric Herr, a volunteer with Search and Support San Antonio, does search and rescue work on the banks of the Guadalupe River in Ingram, Texas, Tuesday, July 8, 2025, after the Fourth of July flood. (Jay Janner/Austin American-Statesman via AP)

“This tragedy, as horrific as it is, could have been so much worse,” Lamb told a news conference, crediting first responders and volunteers with saving lives and knocking on doors to evacuate residents during the flash floods on the July Fourth holiday.

More than 160 people still are believed to be missing, and at least 118 have died in the floods that laid waste to the Hill Country region of Texas. The large number of missing people suggests that the full extent of the catastrophe is still unclear five days after the disaster.

The floods are now the deadliest from inland flooding in the U.S. since 1976, when Colorado’s Big Thompson Canyon flooded, killing 144 people, said Bob Henson, a meteorologist with Yale Climate Connections.

Related Stories
What to know about the Texas flash floods and the rising death toll
What to know about the Texas flash floods and the rising death toll
Timeline raises questions over how Texas officials handled warnings before July 4 flood
Timeline raises questions over how Texas officials handled warnings before July 4 flood
Risk of life-threatening flooding still high in Texas, death toll tops 80
Risk of life-threatening flooding still high in Texas, death toll tops 80
Crews used backhoes and their bare hands Wednesday to dig through piles of debris that stretched for miles along the Guadalupe River in the search of missing people.


“We will not stop until every missing person is accounted for,” Gov. Greg Abbott said Tuesday. “Know this also: There very likely could be more added to that list.”

Officials face backlash for lack of preparations and warnings
Public officials in the area have come under repeated criticism amid questions about the timeline of what happened and why widespread warnings were not sounded and more preparations were not made.

“Those questions are going to be answered,” Kerr County Sheriff Larry Leitha said. “I believe those questions need to be answered, to the families of the loved ones, to the public.”

But he said the priority for now is recovering victims. “We’re not running. We’re not going to hide from anything,” the sheriff said.

Texas Gov. Greg Abbott speaks during a press conference on Tuesday, July 8, 2025, after touring damage from flash flooding in Hunt, Texas. (AP Photo/Eli Hartman)
Texas Gov. Greg Abbott speaks during a press conference on Tuesday, July 8, 2025, after touring damage from flash flooding in Hunt, Texas. (AP Photo/Eli Hartman)

The governor called on state lawmakers to approve new flood warning systems and strengthen emergency communications in flood prone areas throughout the state when the Legislature meets in a special session that Abbott had already called to address other issues starting July 21. Abbott also called on lawmakers to provide financial relief for response and recovery efforts from the storms.

“We must ensure better preparation for such events in the future,” Abbott said in a statement.

Local leaders have talked for years about the need for a flood warning system, but concerns about costs and noise led to missed opportunities to put up sirens.

Raymond Howard, a city council member in Ingram, said it was “unfathomable” that county officials did not act.

“This is lives. This is families,” he said. “This is heartbreaking.”

Number of missing has soared
A day earlier, the governor announced that about 160 people have been reported missing in Kerr County, where searchers already have found more than 90 bodies.



Officials have been seeking more information about those who were in the Hill Country, a popular tourist destination, during the holiday weekend but did not register at a camp or a hotel and may have been in the area without many people knowing, Abbott said.

The riverbanks and hills of Kerr County are filled with vacation cabins, youth camps and campgrounds, including Camp Mystic, the century-old all-girls Christian summer camp where at least 27 campers and counselors died. Officials said five campers and one counselor have still not been found.

Just two days before the flooding, Texas inspectors signed off on the camp’s emergency planning. But five years of inspection reports released to The Associated Press did not provide any details about how campers would be evacuated.

Challenging search for the dead
With almost no hope of finding anyone alive, search crews and volunteers say they are focused on bringing the families of the missing some closure.

Crews fanned out in air boats, helicopters and on horseback. They used excavators and their hands, going through layer by layer, with search dogs sniffing for any sign of buried bodies.

They looked in trees and in the mounds below their feet. They searched inside crumpled pickup trucks and cars, painting them with a large X, much like those marked on homes after a hurricane.

More than 2,000 volunteers have offered to lend a hand in Kerr County alone, the sheriff said.

How long the search will continue was impossible to predict given the number of people unaccounted for and the miles to cover.

Shannon Ament wore knee-high rubber boots and black gloves as she rummaged through debris in front of her rental property in Kerr County. A high school soccer coach is one of the many people she knows who are still missing.

“We need support. I’m not going to say thoughts and prayers because I’m sick of that,” she said. “We don’t need to be blamed for who voted for who. This was a freak of nature — a freak event.”

Disaster relief chaplain Sandi Gilmer, left, and Dan Beazley search for a permanent place for Beazley's large cross in memory of flooding victims on Tuesday, July 8, 2025, near Camp Mystic in Hunt, Texas, after a flash flood swept through the area. (AP Photo/Eli Hartman)
Disaster relief chaplain Sandi Gilmer, left, and Dan Beazley search for a permanent place for Beazley’s large cross in memory of flooding victims on Tuesday, July 8, 2025, near Camp Mystic in Hunt, Texas, after a flash flood swept through the area. (AP Photo/Eli Hartman)

Trump plans to survey damage Friday
President Donald Trump has pledged to provide whatever relief Texas needs to recover. He plans to visit the state Friday.

Polls taken before the floods show Americans largely believe the federal government should play a major role in preparing for and responding to natural disasters.

Catastrophic flooding is a growing worry. On Tuesday, a deluge in New Mexico triggered flash floods that killed three people.

Although it’s difficult to attribute a single weather event to climate change, experts say a warming atmosphere and oceans make these type of storms more likely.


    """
    
    # Split into sentences (you might want to use a more sophisticated sentence splitter)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    print(f"Processing {len(sentences)} sentences...")
    for i, sentence in enumerate(sentences):
        print(f"{i}: {sentence}")
    
    # Clustering parameters
    weights = {"single": 0.7, "context": 0.3}  # Adjusted weights
    
    # Perform clustering
    labels, representative_indices = cluster_sentences(
        sentences, 
        weights, 
        context=True, 
        context_len=2,  # Reduced context length for small dataset
        preprocess=True
    )
    
    print(f"\nCluster Labels: {labels}")
    print(f"Representative Indices: {representative_indices}")
    
    # Print clusters
    print("\nClusters:")
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_sent_list = [sentences[i] for i in range(len(sentences)) if labels[i] == label]
        if label == -1:
            print(f"Noise (unclustered): {cluster_sent_list}")
        else:
            rep_idx = representative_indices[label] if label < len(representative_indices) else None
            rep_sentence = sentences[rep_idx] if rep_idx is not None else "None"
            print(f"Cluster {label}: {cluster_sent_list}")
            print(f"  Representative: {rep_sentence}")
    
    # Visualize the clusters
    if len(sentences) > 1:
        embeddings = model.encode(sentences, show_progress_bar=True)
        visualize_clusters(embeddings, labels)

def cluster_text(text, weights=None, context=True, context_len=1, preprocess=True, 
                 min_cluster_size=2, sentence_splitter=None):
    """
    High-level function to cluster text into semantic groups.
    
    Args:
        text: Input text to cluster
        weights: Dict with 'single' and 'context' weights (default: balanced)
        context: Whether to use context embeddings
        context_len: Number of sentences before/after to include in context
        preprocess: Whether to preprocess text
        min_cluster_size: Minimum size for clusters
        sentence_splitter: Custom function to split text into sentences
    
    Returns:
        dict: Results with clusters, labels, and representatives
    """
    if weights is None:
        weights = {"single": 0.7, "context": 0.3}
    
    # Split text into sentences
    if sentence_splitter:
        sentences = sentence_splitter(text)
    else:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if len(sentences) < 2:
        return {
            "sentences": sentences,
            "labels": np.array([0] if sentences else []),
            "clusters": {0: sentences} if sentences else {},
            "representatives": sentences[:1] if sentences else []
        }
    
    # Perform clustering
    labels, rep_indices = cluster_sentences(sentences, weights, context, context_len, preprocess)
    
    # Organize results
    clusters = {}
    unique_labels = np.unique(labels)
    representatives = []
    
    for label in unique_labels:
        cluster_sents = [sentences[i] for i in range(len(sentences)) if labels[i] == label]
        clusters[label] = cluster_sents
        
        if label != -1 and rep_indices:
            # Find the representative for this cluster
            cluster_rep_indices = [idx for idx in rep_indices if labels[idx] == label]
            if cluster_rep_indices:
                representatives.append(sentences[cluster_rep_indices[0]])
    
    return {
        "sentences": sentences,
        "labels": labels,
        "clusters": clusters,
        "representatives": representatives,
        "num_clusters": len([l for l in unique_labels if l != -1])
    }

if __name__ == "__main__":
    # Test the high-level function first
    test_text = """
    Machine learning is changing the world. Artificial intelligence is the future.
    Dogs are loyal pets. Cats are independent animals. 
    Python is a programming language. JavaScript is used for web development.
    Climate change is a serious issue. Global warming affects everyone.
    """
    
    print("=== Testing High-Level Function ===")
    result = cluster_text(test_text)
    print(f"Found {result['num_clusters']} clusters from {len(result['sentences'])} sentences")
    
    for label, cluster_sents in result['clusters'].items():
        if label == -1:
            print(f"Noise: {cluster_sents}")
        else:
            print(f"Cluster {label}: {cluster_sents}")
    
    print(f"Representatives: {result['representatives']}")
    print("\n" + "="*50 + "\n")
    
    # Run the detailed main function
    main()