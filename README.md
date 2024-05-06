This repository contains my coursework assignments for the CSE 5334 Data Mining course, part of my Master of Science in Computer Science program. This repository contains a collection of assignments demonstrating various concepts and techniques in Data Mining.

Each folder in this repository corresponds to a specific assignment from the Data Mining course. The contents include source code, outputs, and documentation for each task.

## Assignments Overview

### Programming Assignment 1:

-   **Focus**: tokenization, stop word removal, stemming, TF, DF, IDF, TF-IDF, cosine similarity
-   **Description**: This assignment focuses on the development of a basic search engine that reads a corpus and produces TF-IDF vectors for documents. The tasks are executed on a corpus of 30 inaugural addresses by different US presidents, provided in the format of text files. Key tasks include:
    -	Text Processing: Implementation of text processing techniques such as tokenization, stop word removal, and stemming using the NLTK library.
    -	TF-IDF Computation: Generating TF-IDF vectors for each document in the corpus to capture the importance of terms within documents.
    -	Cosine Similarity: Computing cosine similarity to identify the document most relevant to a given query based on similarity scores.
    -	Search Engine Simulation: Using the processed data to simulate a search engine that can return the document most relevant to a given search query.
    -	Efficient Query Handling: Ensuring the search engine can handle queries efficiently, demonstrating the practical application of information retrieval concepts learned in class.

### Programming Assignment 2:

- 	**Focus**: Data preprocessing, feature selection, classification, model evaluation
-	**Description**: This assignment focuses on applying data mining techniques to a dataset to analyze NBA player statistics and predict player roles. Key components include:
	-	Data Preprocessing: Handling missing values, normalizing datasets, and preparing data for analysis.
	-	Feature Selection: Identifying and selecting the most relevant features based on statistical analysis for use in modeling.
	-	Classification: Select the best classification models to accurately predict player positions. Models compared were: SVM, k-Nearest Neighbors, Decision Trees, and Neural Networks.
	-	Model Evaluation: Evaluating model performance using cross-validation and correlation matrix to identify which classes the model had issues classifying correctly.
	-	10-Fold Stratified Cross-Validation: Employing cross-validation techniques to assess the robustness of the models across different subsets of the dataset, ensuring that the results are generalizable.
	
### Programming Assignment 3:

-   **Focus**: K-means clustering, data normalization, error analysis
-   **Description**: This assignment concentrates on implementing and evaluating the K-means clustering algorithm with various configurations of the number of clusters (k). The assignment tasks are executed on multiple datasets. Key tasks include:
    -   Data Normalization: Standardizing datasets to ensure that features contribute equally to the analysis.
    -   K-means Clustering: Implementation of the K-means clustering algorithm from scratch, including initialization, assignment, and update steps.
    -   Error Analysis: Calculating the Sum of Squared Errors (SSE) after 20 iterations for different values of k to assess the performance of the clustering.
    -   Visualization: Creating plots to visualize the clustering results for each k and plotting SSE against k to find the optimal number of clusters.
    -   Efficient Clustering: Ensuring the algorithm handles data efficiently and scales well with different sizes of datasets and configurations.