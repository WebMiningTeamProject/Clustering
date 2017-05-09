import numpy as np
from wordcloud import WordCloud
#NOTE: pip install wordcloud (C++ Build Tools nötig -> http://landinghub.visualstudio.com/visual-cpp-build-tools (vs2015 reicht aus))
#NOTE: wenn _imaging Probleme macht: auf pillow 4.1.1 upgraden! -> pip install -U pillow  (in anconda noch nicht enthalten)
import matplotlib.pyplot as plt
import os, shutil
from dbhandler import DatabaseHandler


def print_clusters(cluster_assignments, topics):
    print("Topic Overview:")
    clusters_unique, cluster_counts = getUniquesAndCounts(cluster_assignments, topics)
    for c_idx, cluster in enumerate(clusters_unique):
        print("Topic ", c_idx,
              " count: ", cluster_counts[c_idx])
        #print("top terms: ", topics[c_idx]["terms"])
        if "avg_weight" in topics[c_idx] and "article_ratio" in topics[c_idx]:
            print(getKPIString(topics[c_idx]))


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


def storeClustersToDB(cluster_assignments, topics, source_uris):
    dbh = DatabaseHandler()

    # 1) store topics  (table: topic_id(=idx), topTerms (comma seperated)
    # 1.1 Drop Content - realized via drop and create to ensure structure changes take effect
    sql = "DROP TABLE IF EXISTS Cluster"
    dbh.execute(sql)

    sql = """
        CREATE TABLE IF NOT EXISTS Cluster (
            cluster_id  VARCHAR(3) PRIMARY KEY,
            terms       TEXT(500),
            kpis        TEXT(500),
            invalid     CHAR(1),
            cluster_name VARCHAR(100)
        )        
    """
    dbh.execute(sql)

    # 1.2 fill with new topics
    tuples = []
    for topic_idx, topic in topics.items():
        invalid = "X"
        if topics[topic_idx]["keep"]:
            invalid = ""
        tuples.append("(" + str(topic_idx)
                      + ",'" + ",".join(topic["terms"]) + "','"
                      + getKPIString(topic) + "','"
                      + invalid + "','"
                      + getClusterName(topic["terms"]) + "')")
    sql = "INSERT INTO Cluster (cluster_id, terms, kpis, invalid) VALUES "
    sql += ",".join(tuples)
    dbh.execute(sql)

    print("DB: Stored %s cluster" % len(tuples))

    # 2) store assignment (table: article_url, topic_id, weight, rank)
    # 2.1 drop content (same approach as above)
    sql = "DROP TABLE IF EXISTS ClusterAssignment"
    dbh.execute(sql)

    sql = """
            CREATE TABLE IF NOT EXISTS ClusterAssignment (
                source_uri  VARCHAR(255) NOT NULL,
                cluster_id  VARCHAR(3) NOT NULL,
                weight      FLOAT(10,10),
                rank        SMALLINT,
                PRIMARY KEY(source_uri, cluster_id)
            )        
        """
    dbh.execute(sql)

    # 2.2 store top 5 assignments
    tuples = []
    setRemovedClustersToNegative(cluster_assignments, topics)
    if len(cluster_assignments.shape) > 1:  # soft clustering
        assignment_sorter = np.argsort(cluster_assignments, axis=1)[:,-5:]  # top 5 topics (ascending sorted)
        for i, weights in enumerate(cluster_assignments):
            for j, cluster_idx in enumerate(assignment_sorter[i,:][::-1]): #get descending sort for article
                if weights[cluster_idx] > 0:
                    tuples.append('("' + source_uris[i].replace('"', "") + '",'
                                  + str(cluster_idx) + ","
                                  + str(weights[cluster_idx]) + ","
                                  + str(j + 1) + ")")
    else: # hard clustering
        for i, cluster_idx in enumerate(cluster_assignments):
                if cluster_idx >= 0: # valid cluster
                    tuples.append("(" + source_uris[i] + ","
                                  + str(cluster_idx) + ","
                                  + "1,1)") # weight and rank are 1 always for hard clustering

    sql = "INSERT INTO ClusterAssignment (source_uri, cluster_id, weight, rank) VALUES "
    sql += ",".join(tuples)
    dbh.execute(sql)

    print("DB: Stored %s assignments" % len(tuples))


# --- Helper Methods
def getUniquesAndCounts(cluster_assignments, topics):
    clusters_unique = topics.keys()
    if len(cluster_assignments.shape) > 1:  # soft clustering
        cluster_counts = np.count_nonzero(cluster_assignments, axis=0)
    else:  # hard clustering
        cluster_counts = np.unique(cluster_assignments, return_counts=True)[1]
    return clusters_unique, cluster_counts


def filterTopics(topics): #inplace
    count = 0
    for topic_idx, topic in topics.items():
        keep = False
        if 0.04 < topic["article_ratio"] < 0.5 and (topic["median_weight"] > 0.005 or topic["top_ratio"] > 0.3):
            keep = True
            count += 1
        topics[topic_idx]["keep"] = keep
        # if(not keep):
        #     cluster_assignments[:,topic_idx] = 0
    print("Filtered Clusters, %s remaining" % count)


def setRemovedClustersToNegative(cluster_assignments, topics): #inplace
    if len(cluster_assignments.shape) > 1:  # soft clustering
        for topic_idx, topic in topics.items():
            if not topics[topic_idx]["keep"]:
                cluster_assignments[:,topic_idx] = -1
    else:
        for topic_idx, topic in topics.items():
            if not topics[topic_idx]["keep"]:
                cluster_assignments[cluster_assignments == topic_idx] = -1

    print("Removed invalid clusters, %s assignments remaining" % len(cluster_assignments[cluster_assignments > 0]))

def getKPIString(topic):
    # string = "KPIs: Avg. Weight = " + str(topic["avg_weight"]) +\
    #          ", Article Ratio = " + str(topic["article_ratio"]) +\
    #          ", Median Weight = " + str(topic["median_weight"]) +\
    #          ", Std. Weight = " + str(topic["std_weight"]) +\
    #          ", Top Ratio = " + str(topic["top_ratio"])
    string = "KPIs: Avg. Weight = %s, Article Ratio = %s, Median Weight = %s, Std. Weight = %s,Top Ratio = %s""" \
             % (topic["avg_weight"],
                topic["article_ratio"],
                topic["median_weight"],
                topic["std_weight"],
                topic["top_ratio"])

    return string


def getClusterName(topic, max_terms=3):
    all_terms = np.array(topic["terms"])
    name_terms = []
    #prio: bigrams
    search_range = max_terms*2
    if search_range >= len(all_terms):
        search_range = len(all_terms) - 1
    for term in all_terms[:search_range]:
        if " " in term and len(name_terms) < max_terms: #bigram!
            name_terms.append(term) # init with first term
    if len(name_terms) < max_terms: #add unigrams
        for term in all_terms[:search_range]:
            if " " not in term and len(name_terms) < max_terms and not any(term in s for s in name_terms): #unigram
                name_terms.append(term)

    return ", ".join(name_terms)



