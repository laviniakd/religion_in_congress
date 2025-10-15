# reimplements the phrase graph construction and partition from the paper "Tracking the Evolution of Memes in Multi-Relational Social Networks" by Parag Singla and Piyush Rai
# https://www.aaai.org/Papers/AAAI/2009/AAAI09-118.pdf
# the "find connected components" function is used to find the phrase clusters

import networkx as nx
import nltk
from collections import Counter
import rapidfuzz.distance as distance
from collections import defaultdict

# Download the NLTK data
nltk.download('punkt')

def tokenize(phrase):
    return nltk.word_tokenize(phrase)

# Function to check k-word overlap
def has_k_word_overlap(p_tokens, q_tokens, k):
    p_length = len(p_tokens)
    q_length = len(q_tokens)
    
    for i in range(p_length - k + 1):
        p_k_sub = p_tokens[i:i+k]
        for j in range(q_length - k + 1):
            q_k_sub = q_tokens[j:j+k]
            if p_k_sub == q_k_sub:
                return True
    return False

# builds the phrase graph with edit distance and k word overlap heuristics
def build_phrase_graph(phrases, delta=1, k=10):
    G = nx.DiGraph()
    
    # Tokenize phrases
    tokenized_phrases = {phrase: tokenize(phrase) for phrase in phrases}
    
    # Add nodes
    for phrase in phrases:
        G.add_node(phrase)
    
    # Add edges
    for p in phrases:
        for q in phrases:
            if len(p) < len(q):
                edit_dist = distance.Levenshtein.distance(tokenized_phrases[p], tokenized_phrases[q])
                # the two heuristics are edit distance and word overlap
                if edit_dist < delta or has_k_word_overlap(tokenized_phrases[p], tokenized_phrases[q], k):
                    G.add_edge(p, q)
    
    return G

# edge weights are edit distance (token-wise)
def add_edge_weights(G, phrases):
    phrase_freq = Counter(phrases)
    
    for p, q in G.edges():
        edit_dist = distance.Levenshtein.distance(tokenize(p), tokenize(q))
        weight = (1 / (1 + edit_dist)) * phrase_freq[q]
        G[p][q]['weight'] = weight

    return G


def find_root_nodes(G):
    return [node for node in G.nodes() if G.in_degree(node) == 0]


# Heuristic-based DAG partitioning -- from paper
def dag_partitioning(G):
    # Find root nodes
    root_nodes = find_root_nodes(G)
    
    # Initialize clusters
    clusters = {root: [root] for root in root_nodes}
    node_to_cluster = {root: root for root in root_nodes}
    
    # Process nodes in topological order
    for node in nx.topological_sort(G):
        if node in root_nodes:
            continue
        
        # Determine which cluster to assign the node to
        max_edges = 0
        best_cluster = None
        cluster_edges = defaultdict(int)
        
        for pred in G.predecessors(node):
            if pred in node_to_cluster:
                cluster = node_to_cluster[pred]
                cluster_edges[cluster] += 1
                if cluster_edges[cluster] > max_edges:
                    max_edges = cluster_edges[cluster]
                    best_cluster = cluster
        
        if best_cluster is not None:
            clusters[best_cluster].append(node)
            node_to_cluster[node] = best_cluster

    return clusters

# Function to find connected components in the graph
# L: minimum length of phrase
# M: minimum frequency of phrases in corpus
def find_connected_components(phrases, L=4, M=10, delta=1, k=10, out_path='/data/laviniad/references/partitions.txt'):
    phrase_counts = Counter(phrases)
    phrases = [phrase for phrase in phrases if len(phrase.split()) > L and phrase_counts[phrase] >= M]

    G = build_phrase_graph(phrases, delta, k)
    G = add_edge_weights(G, phrases)
    
    # partition G into connected components
    partitions = dag_partitioning(G)
    partitions = [sorted(list(partition)) for partition in partitions.values()]

    # save partitions
    with open(out_path, 'w') as f:
        for partition in partitions:
            f.write(','.join(partition) + '\n')

    connected_components = list(nx.strongly_connected_components(G))
    connected_components = [sorted(list(component)) for component in connected_components]

    
    return connected_components, partitions
