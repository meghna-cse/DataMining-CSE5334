"""
Author: Meghna Jaglan
Student ID: 1002053631
Course: CSE 5334 Data Mining
Description: Solution for P3 assignment
"""

#Imported all the required libraries
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler

#------------------------------------------------------------
# Utility functions for data management and preprocessing
#------------------------------------------------------------

# Function to load the dataset from a specified file path and return it as a numpy array.
def load_data(file_path):
    data_frame = pd.read_csv(file_path, delim_whitespace=True, header=None)
    return data_frame.values


# Function to standardize the dataset by applying z-score normalization to the 
# feature columns excluding the last label column.
def standardize_data(data):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data[:, :-1])        # last column is a label
    return np.hstack((features_scaled, data[:, -1:]))


#------------------------------------------------------------
# Supporting functions for K-means algorithm
#------------------------------------------------------------

# Function to randomly select 'k' initial centroids from the data
def select_initial_centroids(data, k):
    indices = random.sample(range(len(data)), k)        # Randomly picks 'k' unique indices for initial centroids
    centroids = [data[i] for i in indices]              # Retrieves and returns the data points at these indices as initial centroids
    return centroids


# Function to calculate Euclidean distance between two points
def calculate_distance(a, b):
    return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))


# Function to assign each data point to the nearest centroid, forming clusters
def form_clusters(data, centroids):
    cluster_assignment = {}                             # Dictionary to store the cluster assignments
    for point in data:                                  # Iterates over each data point

        # Finds the centroid closest to the current point
        closest_centroid = min([(i, calculate_distance(point, centroid)) for i,
                                 centroid in enumerate(centroids)],
                                key=lambda x: x[1])[0]
        
        # Add the point to the corresponding cluster
        if closest_centroid not in cluster_assignment:  
            cluster_assignment[closest_centroid] = []
        cluster_assignment[closest_centroid].append(point)
    return cluster_assignment


# Function to recalculate centroids based on the mean of the points in each cluster
def update_centroids(cluster_data):
    new_centroids = []
    for key in sorted(cluster_data.keys()):
        new_centroids.append(np.mean(cluster_data[key], axis=0).tolist())
    return new_centroids


# Function to compute the total error as the sum of squared distances from each point to its cluster's centroid.
def compute_error(clusters, centroids):
    total_error = 0                                     # Initializes the error as zero
    for key, points in clusters.items():                # Sum of squared distances from each point to its cluster centroid
        centroid = np.array(centroids[key])
        total_error += np.sum((np.array(points) - centroid)**2)
    return total_error


#------------------------------------------------------------
# K-means clustering algorithm implementation
#------------------------------------------------------------

# Implementation of the K-means algorithm
def perform_k_means(data, k, max_iterations=300):
    centroids = select_initial_centroids(data, k)       # Initial selection of centroids
    error_at_iteration_twenty = None                    # Placeholder for error at 20th iteration
    for i in range(max_iterations):
        
        clusters = form_clusters(data, centroids)       # Form clusters
        updated_centroids = update_centroids(clusters)  # Recalculate centroids
        
        if i == 19:                                     # Record error at the 20th iteration
            error_at_iteration_twenty = compute_error(clusters, centroids)
        
        # Check for convergence (if centroids do not change)
        if np.allclose(centroids, updated_centroids, rtol=1e-6):
            if error_at_iteration_twenty is None:
                error_at_iteration_twenty = compute_error(clusters, centroids)
            break
        centroids = updated_centroids

    # If convergence happens before 20 iterations
    if error_at_iteration_twenty is None:
        error_at_iteration_twenty = compute_error(clusters, centroids)
    return centroids, clusters, error_at_iteration_twenty


#------------------------------------------------------------------------------
# Visualization functions
#------------------------------------------------------------------------------

# Plots the error values across different values of k
def visualize_errors(errors, k_range):
    plt.plot(k_range, errors)
    plt.xlabel('Number of Clusters K')
    plt.ylabel('Error Value')
    plt.title('Cluster Error Analysis')
    plt.show()


# Main
if __name__ == "__main__":
    
    dataset_path = sys.argv[1]                          # Retrieves dataset path from command line argument
    dataset_raw = load_data(dataset_path)               # Load data
    processed_data = standardize_data(dataset_raw)      # Standardize data
    errors = []                                         # To store the SSE for each k
    k_range = list(range(2, 11))

    # Perform K-means clustering for different values of k
    for k in k_range:
        centroids, clusters, error_at_20 = perform_k_means(processed_data, k)
        errors.append(error_at_20)
        print(f"For k={k} After 20 iterations: Error = {error_at_20:.4f}")

    # Visualize errors for different values of k
    visualize_errors(errors, k_range)


# Command to run the script: python P3_MeghnaJaglan_1002053631.py <path_to_dataset>
# Example commands:
# python P3_MeghnaJaglan_1002053631.py .\UCI_datasets\satellite_training.txt
# python P3_MeghnaJaglan_1002053631.py .\UCI_datasets\pendigits_training.txt
# python P3_MeghnaJaglan_1002053631.py .\UCI_datasets\yeast_training.txt