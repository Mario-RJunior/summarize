# -*- coding: utf-8 -*-
import re
import nltk
import string
import numpy as np
import networkx as nx
from nltk.cluster.util import cosine_distance
# import streamlit as st
from docx import Document
from unicodedata import normalize
import time

# nltk.download('punkt')
# nltk.download('stopwords')

# Formata o estilo do texto
def format_text(text):
    t0 = time.time()
    
    original_text = re.sub(r'\s+', ' ', text).replace(u'\xa0', u' ')
    
    # original_text = re.sub(r'•', ' ', text)

    print(f'Função: format_text; \nTempo: {time.time() - t0} segundos')
    return original_text

# Retorna texto formatado em letras minúsculas, sem stopwords e pontuação
def preprocess(text):
    t0 = time.time()

    formatted_text = format_text(text)
    formatted_text = text.lower()
    tokens = []
    for token in nltk.word_tokenize(formatted_text):
        tokens.append(token)
    tokens = [word for word in tokens if word not in stopwords and word not in string.punctuation]
    formatted_text = ' '.join(element for element in tokens)

    print(f'Função: preprocess; \nTempo: {time.time() - t0} segundos')

    return formatted_text

def calculate_sentence_similarity(sentence1, sentence2):
    #t0 = time.time()

    words1 = [word for word in nltk.word_tokenize(sentence1)]
    words2 = [word for word in nltk.word_tokenize(sentence2)]

    all_words = list(set(words1 + words2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for word in words1: # Bag of words
        vector1[all_words.index(word)] += 1
    for word in words2:
        vector2[all_words.index(word)] += 1

    #print(f'Função: calculate_sentence_similarity; \nTempo: {time.time() - t0} segundos')

  
    return 1 - cosine_distance(vector1, vector2)

def calculate_similarity_matrix(sentences):
    t0 = time.time()

    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    #print(similarity_matrix)
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            similarity_matrix[i][j] = calculate_sentence_similarity(sentences[i], sentences[j])

    print(f'Função: calculate_similarity_matrix; \nTempo: {time.time() - t0} segundos')

    return similarity_matrix

def summarize(text, number_of_sentences, percentage = 0):
    t0 = time.time()

    original_sentences = [sentence for sentence in nltk.sent_tokenize(text)]
    formatted_sentences = [preprocess(original_sentence) for original_sentence in original_sentences]
    similarity_matrix = calculate_similarity_matrix(formatted_sentences)

    similarity_graph = nx.from_numpy_array(similarity_matrix)

    scores = nx.pagerank(similarity_graph, max_iter=600)
    ordered_scores = sorted(((scores[i], score) for i, score in enumerate(original_sentences)), reverse=True)

    if percentage > 0:
        number_of_sentences = int(len(formatted_sentences) * percentage)

    best_sentences = []
    for sentence in range(number_of_sentences):
        best_sentences.append(ordered_scores[sentence][1])
    
    print(f'Função: summarize; \nTempo: {time.time() - t0} segundos')

  
    return original_sentences, best_sentences, ordered_scores

def get_text(filename):
    t0 = time.time()

    doc = Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    
    print(f'Função: get_text; \nTempo: {time.time() - t0} segundos')

    
    return '\n'.join(fullText)

def generate_summary(origina_text, best_text):
    t0 = time.time()

    text_summ = [frase for frase in origina_text if frase in best_text]
    s = '. '.join(text_summ)

    print(f'Função: generate_summary; \nTempo: {time.time() - t0} segundos')

    return s
    

if __name__ == '__main__':
    t0 = time.time()

    stopwords_en = nltk.corpus.stopwords.words('english')
    stopwords_pt = nltk.corpus.stopwords.words('portuguese')

    stopwords = stopwords_en.copy()
    stopwords.extend(stopwords_pt)
    np.seterr(divide='ignore', invalid='ignore')

    original_text = """
    There are two general approaches to automatic summarization: extraction and abstraction.

    Extraction-based summarization
    Here, content is extracted from the original data, but the extracted content is not modified in any way. Examples of extracted content include key-phrases that can be used to "tag" or index a text document, or key sentences (including headings) that collectively comprise an abstract, and representative images or video segments, as stated above. For text, extraction is analogous to the process of skimming, where the summary (if available), headings and subheadings, figures, the first and last paragraphs of a section, and optionally the first and last sentences in a paragraph are read before one chooses to read the entire document in detail.[6] Other examples of extraction that include key sequences of text in terms of clinical relevance (including patient/problem, intervention, and outcome).[7]

    Abstraction-based summarization
    This has been applied mainly for text. Abstractive methods build an internal semantic representation of the original content, and then use this representation to create a summary that is closer to what a human might express. Abstraction may transform the extracted content by paraphrasing sections of the source document, to condense a text more strongly than extraction. Such transformation, however, is computationally much more challenging than extraction, involving both natural language processing and often a deep understanding of the domain of the original text in cases where the original document relates to a special field of knowledge. "Paraphrasing" is even more difficult to apply to image and video, which is why most summarization systems are extractive.

    Aided summarization
    Approaches aimed at higher summarization quality rely on combined software and human effort. In Machine Aided Human Summarization, extractive techniques highlight candidate passages for inclusion (to which the human adds or removes text). In Human Aided Machine Summarization, a human post-processes software output, in the same way that one edits the output of automatic translation by Google Translate."""

    
    text_word = get_text('teste-v2.docx')
    original_sentences, best_sentences, scores = summarize(text_word, 120, 0.3)
    
    summary = generate_summary(original_sentences, best_sentences)
    print(summary)
    print('-' * 50)
    print(f'{time.time() - t0} segundos')
    print(f'{(time.time() - t0) / 60} minutos')
    
    # 162.59371948242188 segundos
    # 2.7099119782447816 minutos

    """
    raise nx.PowerIterationFailedConvergence(max_iter)
networkx.exception.PowerIterationFailedConvergence: (PowerIterationFailedConvergence(...), 'power iteration failed to converge within 100 iterations')
    """