import numpy as np
from wordcloud import WordCloud
#NOTE: pip install wordcloud (C++ Build Tools nötig -> http://landinghub.visualstudio.com/visual-cpp-build-tools (vs2015 reicht aus))
#NOTE: wenn _imaging Probleme macht: auf pillow 4.1.1 upgraden! -> pip install -U pillow  (in anconda noch nicht enthalten)
import matplotlib.pyplot as plt
import os, shutil


def print_clusters(cluster_assignments, topics):
    print("Topic Overview:")
    clusters_unique, cluster_counts = getUniquesAndCounts(cluster_assignments, topics)
    for c_idx, cluster in enumerate(clusters_unique):
        print("Topic ", c_idx,
              " count: ", cluster_counts[c_idx])
        #print("top terms: ", topics[c_idx]["terms"])
        if "avg_weight" in topics[c_idx] and "article_ratio" in topics[c_idx]:
            print("KPIs: Avg. Weight = ",topics[c_idx]["avg_weight"],
                  ", Article Ratio = ", topics[c_idx]["article_ratio"],
                  ", Median Weight = ", topics[c_idx]["median_weight"],
                  ", Std. Weight = ", topics[c_idx]["std_weight"],
                  ", Top Ratio = ", topics[c_idx]["top_ratio"])


def create_wordclouds(cluster_assignments, topics, files_path="files/wordclouds/", prefix="", clear_path=False):
    print("Create WordClouds per topic in ", files_path)
    if clear_path:
        shutil.rmtree(files_path,ignore_errors=True)
    if not os.path.exists(files_path):
        os.makedirs(files_path)
    clusters_unique, cluster_counts = getUniquesAndCounts(cluster_assignments, topics)
    for cluster in clusters_unique:
        topic = topics[cluster]
        #input format for wordcloud
        dic = dict(zip(topic["terms"], topic["weights"]))
        # create it
        wordcloud = WordCloud().generate_from_frequencies(dic)

        #add additional infos:
        # int at least 5%, but at most 50% of topics present AND
        # keep = "N"
        # if 0.04 < topic["article_ratio"] < 0.5 and (topic["median_weight"] > 0.005 or topic["top_ratio"] > 0.3):
        #     keep = "Y"
        add_info = "_" + str(cluster_counts[cluster]) + "_" + str(topic["keep"])

        # save it
        plt.imsave(fname=files_path + prefix + "topic" + str(cluster) + add_info + ".png",
               arr=wordcloud)

def storeDB(dbHandler, cluster_assignments, topics):
    # 1) store topics  (table: topic_id(=idx), topTerms (comma seperated)
    # 1.1 drop content
    #dbHandler.execute(sql)
    # 1.2 fill with new topics

    # 2) store assignment (table: article_url, topic_id, weight, rank)
    # 2.1 drop content
    # 2.2 store top 5 assignments
    pass



# --- Helper Methods
def getUniquesAndCounts(cluster_assignments, topics):
    clusters_unique = topics.keys()
    if (len(cluster_assignments.shape) > 1):  # soft clustering
        cluster_counts = np.count_nonzero(cluster_assignments, axis=0)
    else:  # hard clustering
        cluster_counts = np.unique(cluster_assignments, return_counts=True)[1]
    return clusters_unique, cluster_counts

def filterTopics(topics): #inplace
    for topic_idx, topic in topics.items():
        keep = False
        if 0.04 < topic["article_ratio"] < 0.5 and (topic["median_weight"] > 0.005 or topic["top_ratio"] > 0.3):
            keep = True
        topics[topic_idx]["keep"] = keep
        # if(not keep):
        #     cluster_assignments[:,topic_idx] = 0



