import networkx as nx
import numpy as np
import math
from collections import Counter


def create_graph_features(n_documents, clean_train_documents, unique_words, 
                          sliding_window, train_par, idf_learned):
    
    """
    :param n_documents: number of documents.
    :param clean_train_documents: the collection.
    :param unique_words: list of all the words we found.
    :param sliding_window: window size.
    :param train_par: if true, we are in the training documents.
    :param idf_learned: 
    :return: 1: features, 2: idf values (for the test data), 3: the list of terms
    """
    
    # Array to store features.
    features = np.zeros((n_documents, len(unique_words)))
    
    # Dictionary of each word with a count of that word throughout the collections.
    term_num_docs = {} 
    
    # Dictionary of each word with the idf of that word.
    idf_col = {}  

    # TODO:
    # 1.idf_col:IDF for the collection
    #   if in training phase compute it
    #   else use the one provided
    # 2. term_num_docs : count of the words in the collection
    #   if in training phase populate it
    #   else use the one provided
    
    if train_par:
        for i in range(n_documents):
            
            # word_list_1 = clean_train_documents[i].split(' ')
            # word_list_2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
            word_list_2 = clean_train_documents[i].split(None)
            
            # Count word occurrences through the collection (for idf).
            # Put the count in term_num_docs.
            
            if len(word_list_2) > 1:
                count_words(word_list_2, term_num_docs)
        
        # Calculate the idf for all words.

        for term_x in term_num_docs:
            idf_col[term_x] = math.log10(float(n_documents)/term_num_docs[term_x])            
    
    else:
        # For the testing set
        # Use the existing ones if we are in the test data.
        idf_col = idf_learned 
        term_num_docs = unique_words

    print "Creating the graph of words for each document..."
    total_nodes = 0
    total_edges = 0

    for i in range(n_documents):
        
        print "On en est au document :", i
        
        # word_list_1 = clean_train_documents[i].split(' ')
        # word_list_2 = [string.rstrip(x.lower(), ',.!?;') for x in wordList1]
        word_list_2 = clean_train_documents[i].split(None)

        dg = nx.Graph()

        if len(word_list_2) > 1:
            
            populate_graph(word_list_2, dg, sliding_window)
            dg.remove_edges_from(dg.selfloop_edges())
            centrality = nx.degree_centrality(dg)  # dictionary of centralities (node: degree)

            total_nodes += dg.number_of_nodes()
            total_edges += dg.number_of_edges()

            # For all nodes
            #   If they are in the desired features
            #       compute the TWIDF score and put it in features[i,unique_words.index(g)].
            
            for k, node_term in enumerate(dg.nodes()):
                if node_term in idf_col:
                    features[i, unique_words.index(node_term)] = \
                        centrality[node_term] * idf_col[node_term]

    if train_par:
        nodes_ret = term_num_docs.keys()

        # print "Percentage of features kept:" + str(feature_reduction)
        print "Average number of nodes:" + str(float(total_nodes)/n_documents)
        print "Average number of edges:" + str(float(total_edges)/n_documents)
    else:
        nodes_ret = term_num_docs

    return features, idf_col, nodes_ret
    
    
def populate_graph(word_list, dg, sliding_window):
    
    """
    For each position/word in the word list:
        add the -new- word in the graph
        for all words -forward- within the window size
            add new words as new nodes 
            add edges among all word within the window.
    
    :param word_list: 
    :param dg: 
    :param sliding_window: 
    :return: 
    """
    
    for k, word in enumerate(word_list):
        if not dg.has_node(word):
            dg.add_node(word)

        temp_w = sliding_window
        if k + sliding_window > len(word_list):
            temp_w = len(word_list) - k
            
        for j in xrange(1, temp_w):
            next_word = word_list[k + j]
            dg.add_edge(word, next_word)
          
            
def count_words(word_list, term_num_docs):
    
    """
    Add the terms from the word_list to the term_num_docs dictionary or increase its count.
    
    :param word_list: 
    :param term_num_docs: 
    :return: 
    """

    found = set()

    for k, word in enumerate(word_list):
        if word not in found:
            found.add(word)
            if word in term_num_docs:
                term_num_docs[word] += 1
            else:
                term_num_docs[word] = 1
