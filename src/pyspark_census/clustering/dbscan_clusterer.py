'''
Created on May 30, 2025

@author: kevinrjamesii
'''
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import logging

class DbScanClusterer(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        logging.debug("Initialized DbScan Clusterer")
        
    def plot_dbscan(self, X, features):
        """
        X: numpy array used for clustering (raw, scaled data)
        feature_df: original pandas DataFrame with named features (not PCA)
        """
        # Reduce dimensionality for plotting only
        pca = PCA(n_components=6)
        X_pca = pca.fit_transform(X)
    
        # Try multiple DBSCAN configurations
        params = [
            {"eps": 2.0, "min_samples": 3},
            {"eps": 2.5, "min_samples": 5},
            {"eps": 2.7, "min_samples": 5},
            {"eps": 3.0, "min_samples": 3},
        ]
    
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    
        for i, p in enumerate(params):
            db = DBSCAN(eps=p["eps"], min_samples=p["min_samples"]).fit(X_pca)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
            # Plot DBSCAN results
            palette = sns.color_palette('tab10', n_colors=20)
            sns.scatterplot(
                x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette=palette, ax=axes[i], legend=None
            )
            axes[i].set_title(f"eps={p['eps']}, min_samples={p['min_samples']} → Clusters: {n_clusters}")
            axes[i].set_xlabel("PCA 1")
            axes[i].set_ylabel("PCA 2")
    
            # Profile the clusters (excluding noise, label = -1)
            df_labeled = features.copy()
            df_labeled['cluster'] = labels
            df_filtered = df_labeled[df_labeled['cluster'] != -1]
    
            if df_filtered.empty:
                print(f"⚠️ All points classified as noise for eps={p['eps']}, min_samples={p['min_samples']}")
            else:
                profile = df_filtered.groupby('cluster').mean(numeric_only=True).round(2)
                print(profile)
    
            print(f"\n📊 Cluster profile for DBSCAN (eps={p['eps']}, min_samples={p['min_samples']}):")

    
        plt.suptitle("DBSCAN Clustering Results with Different Hyperparameters", fontsize=16)
        plt.tight_layout()
        plt.show()
  