'''
Created on May 23, 2025

@author: kevinrjamesii
'''

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class KMeansClusterer(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    def plot_kmeans(self, X, num_components=6):
        orig_feature_df = pd.DataFrame(X)
        
        # Clean feature frame
        static_cols = ['unique_id', 'ds', 'year', 'state', 'region', 'NAME', "GEO_ID"]
        feature_cols = [col for col in orig_feature_df.columns if col not in static_cols]
        feature_df = orig_feature_df[feature_cols].fillna(0)
        
        # PCA
        pca = PCA(n_components=num_components)
        X_pca = pca.fit_transform(feature_df)
        pca_df = pd.DataFrame(X_pca, columns=[f'PCA{i+1}' for i in range(num_components)])
        
        # Clustering
        kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(pca_df)
        pca_df['cluster'] = clusters
        
        # Centroid plotting fix
        centroids = kmeans.cluster_centers_
        #x_feature, y_feature = 'B19013_001E', 'B25003_001E'
        x_feature, y_feature = 'PCA1', 'PCA2'
        centroids_x = centroids[:, 0]
        centroids_y = centroids[:, 1]
        
        # Cluster Plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=pca_df, x=x_feature, y=y_feature, hue='cluster', palette='tab10', s=25, alpha=0.6)
        plt.scatter(centroids_x, centroids_y, c='black', s=200, marker='X', label='Centroids')
        plt.title("Clustering in Original Space")
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        # Silhouette Score
        #X = feature_df.drop(columns=['cluster'])
        #labels = feature_df['cluster']
        #score = silhouette_score(X, labels)
        
        # Sample for performance
        sampled_df = pca_df.sample(frac=0.05, random_state=42)
        X = sampled_df.drop(columns=['cluster'])
        labels = sampled_df['cluster']
        
        # Silhouette score
        score = silhouette_score(X, labels)
        print(f"Silhouette Score: {score:.3f}")
        
        # Silhouette Plot
        sample_silhouette_values = silhouette_samples(X, labels)
        y_lower = 10
        plt.figure(figsize=(10, 6))
        for i in range(6):
            cluster_vals = sample_silhouette_values[labels == i]
            cluster_vals.sort()
            size = cluster_vals.shape[0]
            y_upper = y_lower + size
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals)
            plt.text(-0.05, y_lower + size / 2, str(i))
            y_lower = y_upper + 10
        
        plt.axvline(x=score, color="red", linestyle="--")
        plt.xlabel("Silhouette Coefficient")
        plt.ylabel("Cluster")
        plt.title("Silhouette Plot (Sampled)")
        plt.tight_layout()
        plt.show()
        
        # Elbow Method
        inertia = []
        k_range = range(2, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        
        plt.figure(figsize=(8, 6))
        plt.plot(k_range, inertia, marker='o')
        plt.title("Elbow Method for Optimal K")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia (Within-Cluster SSE)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        #X = feature_df.values  # or .to_numpy()
        #self.plot_dbscan(X, feature_df)