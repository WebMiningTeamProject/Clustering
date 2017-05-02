from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
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
    kmeans = KMeans(n_clusters=n_topics, random_state=0).fit(U)
    cluster_assignments = kmeans.labels_
    print("SVD: KMeans: ", len(cluster_assignments), " cluster assignments" )

    clusters_unique = np.unique(cluster_assignments) #sorted unique elements

    # Derive Topics (with top-n words per cluster)
    topics = {}
    # for i, row in enumerate(Vt):
    #    topics[i] = vocab[np.argsort(row)[::-1][:top_n]]
    for c_idx, cluster in enumerate(clusters_unique):
        term_weights = np.asfarray(csr_matrix(matrix)[cluster_assignments == cluster,:].sum(axis=0)).flatten()
        top = term_weights.argsort()[::-1][:top_n]
        top = top[term_weights[top] > 0]
        topics[c_idx] = {"terms": vocab[top],
                         "weights": term_weights[top]}
        #TODO: SVD: KMeans: Topics based on cluster_centers_?
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
    for c_idx, component in enumerate(nmf.components_):
        top = component.argsort()[::-1][:top_n]
        top = top[component[top] > 0]
        topics[c_idx] = {"terms": vocab[top],
                         "weights": component[top]}

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

    return cluster_assignments, topics





