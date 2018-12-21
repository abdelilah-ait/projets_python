# Import necessary libraries
from copy import deepcopy
import matplotlib.cm as cm
from sklearn import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

iris = datasets.load_iris()

X = iris.data
category = iris.target
y = category

def k_mean(data, n_clusters):

    # Number of clusters
    k = n_clusters
    # Number of training data
    n = data.shape[0]
    # Number of features in the data
    c = data.shape[1]

    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    centers = np.random.randn(k, c)*std + mean

    # Plot the data and the centers generated as random
    plt.title('Plot the data and the centers generated as random')
    colors = ['orange', 'blue', 'green']
    for i in range(n):
        plt.scatter(data[i, 0], data[i, 1], s=7, color=colors[int(category[i])])
    plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='g', s=150)
    plt.show()

    centers_old = np.zeros(centers.shape)  # to store old centers
    centers_new = deepcopy(centers)  # Store new centers

    data.shape
    clusters = np.zeros(n)
    distances = np.zeros((n, k))

    error = np.linalg.norm(centers_new - centers_old)

    # When, after an update, the estimate of that center stays the same, exit loop
    while error != 0:
        # Measure the distance to every center
        for i in range(k):
            distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
        # Assign all training data to closest center
        clusters = np.argmin(distances, axis=1)

        centers_old = deepcopy(centers_new)
        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)
        error = np.linalg.norm(centers_new - centers_old)
    centers_new

    # Plot the data and the centers generated as random
    colors = ['orange', 'blue', 'green']
    plt.title('Plot the data and the centers after clustering')
    for i in range(n):
        plt.scatter(data[i, 0], data[i, 1], s=7, color=colors[int(category[i])])
    plt.scatter(centers_new[:, 0], centers_new[:, 1], marker='*', c='g', s=150)
    plt.show()

    
if __name__ == '__main__':

    k_mean(X, 3)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.title('Plot the data and the centers after sklearn Kmeans')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])


    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    #LDA => Cette méthode identifie les composants (c'est-à-dire la combinaison linéaire des variables observées) 
    #qui maximisent la séparation des classes (c'est-à-dire la variance entre classes) 
    #PCA => Vise à trouver des composants qui tiennent compte de la variance maximale 
    #dans les données (y compris la variance d'erreur et la variance intra-variable). Contrairement à LDA, 
    #il ne prend pas en compte l’appartenance à une classe (c’est-à-dire, non supervisé)
        
    plt.show()
    pca = PCA(n_components=2)
    IrisPCA = pca.fit(X).transform(X)
    plt.title('PCA visualisation')
    plt.scatter(IrisPCA[:, 0], IrisPCA[:, 1], c=iris.target, cmap='viridis')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.show()

    lda = LinearDiscriminantAnalysis(n_components=2)
    IrisLDA = lda.fit(X, y).transform(X)
    plt.title('LDA visualisation')
    plt.scatter(IrisLDA[:, 0], IrisLDA[:, 1], c=iris.target, cmap='viridis')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.show()
    
    #algorithme CAH
    
    X = np.array([[5,3],  
        [10,15],
        [15,12],
        [24,10],
        [30,30],
        [85,70],
        [71,80],
        [60,78],
        [70,55],
        [80,91],])

    labels = range(1, 11)  
    plt.figure(figsize=(10, 7))  
    plt.subplots_adjust(bottom=0.1)  
    plt.scatter(X[:,0],X[:,1], label='True Position')

    for label, x, y in zip(labels, X[:, 0], X[:, 1]):  
        plt.annotate(
            label,
            xy=(x, y), xytext=(-3, 3),
            textcoords='offset points', ha='right', va='bottom')
    plt.show()

    linked = linkage(X, 'single')

    labelList = range(1, 11)

    plt.figure(figsize=(10, 7))  
    dendrogram(linked,  
                orientation='top',
                labels=labelList,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.show()  
