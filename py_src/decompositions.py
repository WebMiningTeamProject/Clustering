from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.cluster import silhouette_samples
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix



def calc_svd(matrix,
             vocab,
             svd_k=30,
             n_topics=15,
             top_n = 20):
    #SVD
    print("SVD: Calculating with k=", svd_k, "...")
    U, s, Vt = svds(matrix,k=svd_k)

    print("SVD: Singular Values: ", s[::-1])

    #print("SVD: Top ",top_n, " words for first 10 topics")
    #for i, row in enumerate(Vt[k-10:,:]):
    #    print("Topic ", i, ":")
    #    #print("Row: ", row.shape)
    #    print(vocab[np.argsort(row)[::-1][:top_n]])

    #Use k-means to cluster texts hard (based on U: text-topic assignment)
    print("SVD: KMeans: Calculating ", n_topics, " clusters (topics)...")
    cluster_assignments, topics = applyKMeans(n_topics=n_topics,
                                              raw_data=matrix,
                                              vocab=vocab,
                                              top_n=top_n,
                                              soft_clustering=U)
    print("SVD: KMeans: ", len(cluster_assignments), " cluster assignments")

    # kmeans = KMeans(n_clusters=n_topics, random_state=0).fit(U)
    # cluster_assignments = kmeans.labels_
    # print("SVD: KMeans: ", len(cluster_assignments), " cluster assignments" )
    #
    # clusters_unique = np.unique(cluster_assignments) #sorted unique elements
    #
    # # Derive Topics (with top-n words per cluster)
    # topics = {}
    # # for i, row in enumerate(Vt):
    # #    topics[i] = vocab[np.argsort(row)[::-1][:top_n]]
    # for c_idx, cluster in enumerate(clusters_unique):
    #     term_weights = np.asfarray(csr_matrix(matrix)[cluster_assignments == cluster,:].sum(axis=0)).flatten()
    #     top = term_weights.argsort()[::-1][:top_n]
    #     top = top[term_weights[top] > 0]
    #     topics[c_idx] = {"terms": vocab[top],
    #                      "weights": term_weights[top]}

    calc_metrics(topics=topics, cluster_assignments=cluster_assignments, raw_data=matrix, soft_clustering=False)

    return cluster_assignments, topics

def calc_nmf(matrix,
             vocab,
             n_topics=15,
             top_n=20):

    print("NMF: Calculating ", n_topics, " components (topics)...")
    nmf = NMF(n_components=n_topics,
              random_state=1,
              alpha=.1,
              l1_ratio=.5
              ).fit(matrix)
    print("NMF: reconstruction error:", nmf.reconstruction_err_)

    #soft clustering
    cluster_assignments = nmf.transform(matrix) #samples x components

    #derive topics
    topics = {}
    # #mask_nonzero = cluster_assignments > 0
    # sums = np.sum(cluster_assignments, axis=0)
    # counts = np.count_nonzero(cluster_assignments, axis=0)
    # #medians = np.median(cluster_assignments[mask_nonzero,:],axis=0)
    # #stds = np.std(cluster_assignments[mask_nonzero,:],axis=0)
    # total_count = cluster_assignments.shape[0]
    # most_relevant_topics = np.argsort(cluster_assignments, axis=1)[:, -1]  # just the last column (Ascending sorted!)
    for c_idx, component in enumerate(nmf.components_):
        # determine top terms
        top = component.argsort()[::-1][:top_n]
        top = top[component[top] > 0]

        # # Calc. KPIs
        # avg_weight = sums[c_idx]/counts[c_idx]
        # article_ratio = counts[c_idx]/total_count
        # mask_nonzero = cluster_assignments[:,c_idx] > 0
        # std = np.std(cluster_assignments[mask_nonzero,c_idx])
        # median = np.median(cluster_assignments[mask_nonzero,c_idx])
        # top_ratio = len(np.where(most_relevant_topics == c_idx)[0]) / counts[c_idx]

        # Store
        topics[c_idx] = {"terms": vocab[top],
                         "weights": component[top]
                         # "avg_weight": avg_weight,
                         # "median_weight": median,
                         # "std_weight": std,
                         # "article_ratio": article_ratio,
                         # "top_ratio": top_ratio
                         }

    calc_metrics(topics=topics, cluster_assignments=cluster_assignments, raw_data=matrix)

    return cluster_assignments, topics

def calc_lda(matrix,
             vocab,
             n_topics=15,
             top_n=20):

    print("LDA: Calculating ", n_topics, " topics...")
    lda = LatentDirichletAllocation(n_topics=n_topics,
                                    max_iter=10,
                                    learning_method="batch", #"online"
                                    #learning_offset=50.,
                                    random_state=0
                                    ).fit(matrix)
    #soft clustering
    cluster_assignments = lda.transform(matrix) #samples x topics

    #derive topics
    topics = {}
    for c_idx, component in enumerate(lda.components_):
        top = component.argsort()[::-1][:top_n]
        top = top[component[top] > 0]
        topics[c_idx] = {"terms": vocab[top],
                         "weights": component[top]}

    calc_metrics(topics=topics, cluster_assignments=cluster_assignments, raw_data=matrix)

    return cluster_assignments, topics

def calc_metrics(topics, cluster_assignments, raw_data, soft_clustering=True):
    print("Calculate KPIs...")
    total_count = cluster_assignments.shape[0]

    if soft_clustering:
        sums = np.sum(cluster_assignments, axis=0)
        counts = np.count_nonzero(cluster_assignments, axis=0)
        hard_cluster = np.argsort(cluster_assignments, axis=1)[:, -1]  # just the last column (Ascending sorted!)
        # inter-cluster-sim - intra-cluster-sim / max of both
        silhouette_scores = silhouette_samples(raw_data, hard_cluster)
    else:
        # inter-cluster-sim - intra-cluster-sim / max of both
        silhouette_scores = silhouette_samples(raw_data, cluster_assignments)



    for c_idx, topic in topics.items():
        # Calc. KPIs

        if soft_clustering:
            avg_weight = sums[c_idx]/counts[c_idx]
            article_ratio = counts[c_idx]/total_count
            mask_nonzero = cluster_assignments[:,c_idx] > 0
            std = np.std(cluster_assignments[mask_nonzero,c_idx])
            median = np.median(cluster_assignments[mask_nonzero,c_idx])
            top_ratio = len(np.where(hard_cluster == c_idx)[0]) / counts[c_idx]
            silhouette_score = np.mean(silhouette_scores[hard_cluster == c_idx])

            #Store KPIs
            topic.update({"avg_weight": avg_weight,
                          "median_weight": median,
                          "std_weight": std,
                          "article_ratio": article_ratio,
                          "top_ratio": top_ratio,
                          "silhouette_score": silhouette_score})
        else:
            article_ratio = len(cluster_assignments[cluster_assignments == c_idx]) / total_count
            silhouette_score = np.mean(silhouette_scores[cluster_assignments == c_idx])
            # Store KPIs
            topic.update({"article_ratio": article_ratio,
                          "silhouette_score": silhouette_score})

    return #topics  #inplace update!

def applyKMeans(n_topics,raw_data, vocab, top_n, soft_clustering):
    kmeans = KMeans(n_clusters=n_topics, random_state=0).fit(soft_clustering)
    cluster_assignments = kmeans.labels_
    clusters_unique = np.unique(cluster_assignments)  # sorted unique elements

    # Derive Topics (with top-n words per cluster)
    topics = {}
    # for i, row in enumerate(Vt):
    #    topics[i] = vocab[np.argsort(row)[::-1][:top_n]]
    for c_idx, cluster in enumerate(clusters_unique):
        term_weights = np.asfarray(csr_matrix(raw_data)[cluster_assignments == cluster, :].sum(axis=0)).flatten()
        top = term_weights.argsort()[::-1][:top_n]
        top = top[term_weights[top] > 0]
        topics[c_idx] = {"terms": vocab[top],
                         "weights": term_weights[top]}

    return cluster_assignments, topics








