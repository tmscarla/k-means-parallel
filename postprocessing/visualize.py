import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


if __name__ == '__main__':

    # Load tweets points and topics
    n_dim = 20
    tweets_points = pd.read_csv('../data/tweets_points_{}.csv'.format(n_dim), skiprows=1, header=None)
    topics = pd.read_csv('../data/tweets_topics.csv')

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(tweets_points.values)
    df = pd.DataFrame()
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]

    topic_to_n = {'history':0,
                  'painting':1,
                  'photography':2,
                  'cooking':3,
                  'animal':4,
                  'health':5,
                  'culture':6,
                  'drawing':7,
                  'theatre':8,
                  'music':9,
                  'sport':10,
                  'exercise':11}

    df['topic'] = [topic_to_n[x] for x in topics['topic']]

    print(df)

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df["pca-one"],
        ys=df["pca-two"],
        zs=df["pca-three"],
        c=df["topic"],
        cmap='tab20'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.legend()
    plt.show()
