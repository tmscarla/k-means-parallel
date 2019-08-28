import numpy as np
import pandas as pd



if __name__ == '__main__':
    # tweets = pd.read_csv('data/tweets_points_20.csv')
    res = pd.read_csv('results/memb20dim_cf.csv')
    topics = pd.read_csv('data/tweets_topics.csv')

    print(topics.shape)
    print(res.shape)

    id_to_topic = dict(zip(list(range(len(topics))), topics['topic'].tolist()))

    print('\n----- TOPICS -----')
    print(topics['topic'].value_counts())

    cluster_df = res.groupby('Cluster_id')['Point_id'].apply(list)

    for index, cluster in enumerate(cluster_df):

        print('----- CLUSTER {} -----'.format(index))
        points_labeled = [id_to_topic[p] for p in cluster]
        print(pd.Series(points_labeled).value_counts(normalize=True))
