"""
Author: Meghna J
Course: CSE 5334 Data Mining
Description: Solution for P1 assignment

Pre-requisite:
This script requires the NLTK library and its 'stopwords' corpus. Please ensure NLTK
is installed and the 'stopwords' corpus has been downloaded before running this script.
"""

import nltk
import os
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter


# Initialization
corpusroot = './US_Inaugural_Addresses'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stopWordsFilter = set(stopwords.words('english'))
stemmer = PorterStemmer()
preprocessedDocuments = {}                                     # To store preprocessed content of each document


# ---------------------------------------------------------------------
# Function for preprocessing: Filters text by tokenizing, removing 
# stopwords, and stemming.
# ---------------------------------------------------------------------
def preprocessData(text):
    tokens = tokenizer.tokenize(text.lower())                                           # Tokenize words
    tokens_noStopwords = [word for word in tokens if word not in stopWordsFilter]       # Stopword Removal
    return [stemmer.stem(word) for word in tokens_noStopwords]                          # Stemming


# ---------------------------------------------------------------------
# Function for Normalizing vector: Normalizes a vector to help in 
# calculating cosine similarity.
# ---------------------------------------------------------------------
def normalizeVector(vector):
    normalized = math.sqrt(sum([val**2 for val in vector.values()]))
    return {term: (val / normalized) for term, val in vector.items()}


# ---------------------------------------------------------------------
# Function to compute cosine similarity: Calculates similarity between
# two vectors, heling in identifying relevant documents.
# ---------------------------------------------------------------------
def cosineSimilarity(vectorA, vectorB):
    common = set(vectorA.keys()) & set(vectorB.keys())
    numerator = sum([vectorA[x] * vectorB[x] for x in common])
    sum1 = sum([val**2 for val in vectorA.values()])
    sum2 = sum([val**2 for val in vectorB.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return float(numerator) / denominator if denominator else 0.0


# ---------------------------------------------------------------------
# Document Processing 
# ---------------------------------------------------------------------

# Preprocessing the document
for filename in os.listdir(corpusroot):
    if filename.endswith('.txt'):
        with open(os.path.join(corpusroot, filename), "r", encoding='windows-1252') as file:
            doc = file.read()
            preprocessedDocuments[filename] = preprocessData(doc)
 

# TF-IDF calculation for documents [logarithmic tf, logarithmic idf, cosine normalization]

## Calculate DF
documentFrequency = Counter()
for tokens in preprocessedDocuments.values():                     
    documentFrequency.update(set(tokens))

## Calculate IDF
N = len(preprocessedDocuments)
idf = {term: math.log(N / (dfCount + 1)) + 1 for term, dfCount in documentFrequency.items()}

tf_idf = {}
for doc, tokens in preprocessedDocuments.items():
    termFrequency = Counter(tokens)

    ## Calculate logarithmic TF
    log_tf = {term: (1 + math.log(tfCount)) for term, tfCount in termFrequency.items() if tfCount > 0}
    
    ## Calculate TF-IDF and cosine normalization
    tf_idf_temp = {term: log_tf.get(term, 0) * idf[term] for term in tokens}
    tf_idf[doc] = normalizeVector(tf_idf_temp)



# ---------------------------------------------------------------------
# Query Processing
# ---------------------------------------------------------------------
    
# Function to calculate the Inverse Document Frequency (IDF) for a given token
def getidf(token):
    preprocessedToken = preprocessData(token)
    if not preprocessedToken:                   # If token is filtered out by preprocessing
        return -1
    token = preprocessedToken[0]
    dfCount = documentFrequency.get(token, 0)
    if dfCount == 0:
        return -1
    else:
        return math.log10(N / dfCount)


# Function to retrieve the TF-IDF weight of a specific token within a given document
def getweight(filename, token):
    preprocessedToken = preprocessData(token)
    if not preprocessedToken:                   # If token is filtered out by preprocessing
        return 0
    token = preprocessedToken[0]                # Extract the preprocessed token
    doc_tf_idf = tf_idf.get(filename, {})
    return doc_tf_idf.get(token, 0.0)


# Function to handle queries
def query(inputString):

    # Query processing
    stemmedTokens = preprocessData(inputString)
    queryVec = Counter(stemmedTokens)
    
    # TF-IDF calculation for query [logarithmic tf, no idf, cosine normalization]

    ## Calculate logarithmic TF
    query_tf = {term: (1 + math.log(queryVec[term])) for term in queryVec if queryVec[term] > 0}
    
    ## Calculate cosine normalization
    query_tf_normalized = normalizeVector(query_tf)
    
    matches = {doc: cosineSimilarity(query_tf_normalized, doc_tf_idf) for doc, doc_tf_idf in tf_idf.items()}
    mostRelevantMatch = max(matches, key=matches.get)
    return mostRelevantMatch, matches[mostRelevantMatch]



# ---------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------
print("%.12f" % getidf('children'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('people'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
print("%.12f" % getweight('23_hayes_1877.txt','public'))
print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
print("%.12f" % getweight('05_jefferson_1805.txt','press'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("cuba government"))


'''
Output for the last run:

0.574031267728
0.134698573897
0.029963223377
0.079181246048
0.045757490561
--------------
0.054772720904
0.049971521628
0.057944970001
0.052134256863
0.066195615638
--------------
(03_adams_john_1797.txt, 0.098848956633)
(20_lincoln_1865.txt, 0.218281060473)
(07_madison_1813.txt, 0.158720692335)
(15_polk_1845.txt, 0.111620697507)
(29_mckinley_1901.txt, 0.133838245104)
'''