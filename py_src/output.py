import numpy as np
from wordcloud import WordCloud
#NOTE: pip install wordcloud (C++ Build Tools nötig -> http://landinghub.visualstudio.com/visual-cpp-build-tools (vs2015 reicht aus))
#NOTE: wenn _imaging Probleme macht: auf pillow 4.1.1 upgraden! -> pip install -U pillow  (in anconda noch nicht enthalten)
import matplotlib.pyplot as plt
import os


def print_clusters(cluster_assignments, topics):
    print("Topic Overview:")
    clusters_unique, cluster_counts = getUniquesAndCounts(cluster_assignments, topics)
    for c_idx, cluster in enumerate(clusters_unique):
        print("Topic ", c_idx,
              " count: ", cluster_counts[c_idx])
              #", top terms: ", topics[c_idx]["terms"])


def create_wordclouds(cluster_assignments, topics, files_path="files/wordclouds/", prefix=""):
    print("Create WordClouds per topic in ", files_path)
    if not os.path.exists(files_path):
        os.makedirs(files_path)
    clusters_unique, cluster_counts = getUniquesAndCounts(cluster_assignments, topics)
    for cluster in clusters_unique:
        topic = topics[cluster]
        #input format for wordcloud
        dic = dict(zip(topic["terms"], topic["weights"]))
        # create it
        wordcloud = WordCloud().generate_from_frequencies(dic)
        # save it
        plt.imsave(fname=files_path + prefix + "topic" + str(cluster) + "_" + str(cluster_counts[cluster]) + ".png",
               arr=wordcloud)


def getUniquesAndCounts(cluster_assignments, topics):
    clusters_unique = topics.keys()
    if (len(cluster_assignments.shape) > 1):  # soft clustering
        cluster_counts = np.count_nonzero(cluster_assignments, axis=0)
    else:  # hard clustering
        cluster_counts = np.unique(cluster_assignments, return_counts=True)[1]
    return clusters_unique, cluster_counts
