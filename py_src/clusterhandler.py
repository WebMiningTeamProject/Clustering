from datahandler import DataHandler
import decompositions as dc
import output as out
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.cluster import silhouette_samples
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

class ClusterHandler:
    def __init__(self,
                 n_topics=30,
                 top_n=20, #words per topic
                 soft_offset=5,
                 prefix="t30_12gram_",
                 path="files/wordclouds/"
                 ):
        self.n_topics = n_topics
        self.top_n = top_n
        self.soft_offset = soft_offset
        self.prefix = prefix
        self.path = path

        #self.ngram_range = ngram_range

        #self.datahandler = DataHandler(use_cache=use_cache,ngram_range=ngram_range)
        #self.tfidf, self.tfidf_vocab = self.datahandler.get_tfidf()
        #self.tf, self.tf_vocab = self.datahandler.get_tf()

    ## DECOMPOSITIONS

    def calc_svd(self,
                 matrix,
                 vocab,
                 svd_k=None):
        if svd_k == None:
            svd_k = self.n_topics + self.soft_offset
        # SVD
        print("SVD: Calculating with k=", svd_k, "...")
        U, s, Vt = svds(matrix,
                        k=svd_k)

        #print("SVD: Singular Values: ", s[::-1])
        #
        # for i, row in enumerate(Vt[svd_k-10:,:]):
        #    print("Topic ", i, ":")
        #    #print("Row: ", row.shape)
        #    print(self.tfidf_vocab[np.argsort(row)[::-1][:self.top_n]])

        topics = {}
        for c_idx, component in enumerate(Vt):
            # determine top terms
            top = component.argsort()[::-1][:self.top_n]
            #top = top[component[top] > 0]
            # Store
            topics[c_idx] = {"terms": vocab[top],
                             "weights": component[top]
                             }

        # self.__calc_metrics__(topics=topics,
        #                       cluster_assignments=U,
        #                       raw_data=matrix,
        #                       soft_clustering=True)

        self.__postprocess__(clusters=U,
                             topics=topics,
                             raw_data=matrix,
                             path=self.path + "svd/",
                             prefix=self.prefix,
                             soft_clustering=True)


        # Use k-means to cluster texts hard (based on U: text-topic assignment)
        print("SVD: KMeans: Calculating ", self.n_topics, " clusters (topics)...")
        cluster_assignments, topics = self.__applyKMeans__(raw_data=matrix,
                                                           vocab=vocab,
                                                           soft_clustering=U)
        print("SVD: KMeans: ", len(cluster_assignments), " cluster assignments")

        # self.__calc_metrics__(topics=topics,
        #                       cluster_assignments=cluster_assignments,
        #                       raw_data=matrix,
        #                       soft_clustering=False)

        self.__postprocess__(clusters=cluster_assignments,
                             topics=topics,
                             raw_data=matrix,
                             path=self.path + "svd/kmeans/",
                             prefix=self.prefix,
                             soft_clustering=False)

        return cluster_assignments, topics


    def calc_nmf(self,
                 matrix,
                 vocab,
                 components=None):
        if components is None:
            components = self.n_topics + self.soft_offset

        print("NMF: Calculating ", components, " components (topics)...")
        nmf = NMF(n_components=components,
                  random_state=1,
                  alpha=.1,
                  l1_ratio=.5
                  ).fit(matrix)
        print("NMF: reconstruction error:", nmf.reconstruction_err_)

        # soft clustering
        cluster_assignments = nmf.transform(matrix)  # samples x components

        # derive topics
        topics = {}
        for c_idx, component in enumerate(nmf.components_):
            # determine top terms
            top = component.argsort()[::-1][:self.top_n]
            top = top[component[top] > 0]
            # Store
            topics[c_idx] = {"terms": vocab[top],
                             "weights": component[top]
                             }

        self.__postprocess__(clusters=cluster_assignments,
                             topics=topics,
                             raw_data=matrix,
                             path=self.path + "nmf/",
                             prefix=self.prefix,
                             soft_clustering=True)

        #cluster_assignments = self.__removeInvalid__(cluster_assignments=cluster_assignments, topics=topics)

        print("NMF: KMeans: Calculating ", self.n_topics, " clusters (topics)...")
        cluster_assignments, topics = self.__applyKMeans__(raw_data=matrix,
                                                           vocab=vocab,
                                                           soft_clustering=cluster_assignments)
        print("NMF: KMeans: ", len(cluster_assignments), " cluster assignments")

        self.__postprocess__(clusters=cluster_assignments,
                             topics=topics,
                             raw_data=matrix,
                             path=self.path + "nmf/kmeans/",
                             prefix=self.prefix,
                             soft_clustering=False)


        return cluster_assignments, topics

    def calc_lda(self,
                 matrix,
                 vocab,
                 max_iter=10,
                 n_topics=None):

        if n_topics is None:
            n_topics = self.n_topics + self.soft_offset

        print("LDA: Calculating ", n_topics, " topics with max. ", max_iter ," iterations...")
        lda = LatentDirichletAllocation(n_topics=n_topics,
                                        max_iter=max_iter,
                                        learning_method="batch",  # "online"
                                        # learning_offset=50.,
                                        random_state=0
                                        ).fit(matrix)
        # soft clustering
        cluster_assignments = lda.transform(matrix)  # samples x topics

        # derive topics
        topics = {}
        for c_idx, component in enumerate(lda.components_):
            top = component.argsort()[::-1][:self.top_n]
            top = top[component[top] > 0]
            topics[c_idx] = {"terms": vocab[top],
                             "weights": component[top]}

        self.__postprocess__(clusters=cluster_assignments,
                            topics=topics,
                            raw_data=matrix,
                            path=self.path + "lda/",
                            prefix=self.prefix,
                            soft_clustering=True)

        print("LDA: KMeans: Calculating ", self.n_topics, " clusters (topics)...")
        cluster_assignments, topics = self.__applyKMeans__(raw_data=matrix,
                                                           vocab=vocab,
                                                           soft_clustering=cluster_assignments)
        print("LDA: KMeans: ", len(cluster_assignments), " cluster assignments")

        self.__postprocess__(clusters=cluster_assignments,
                             topics=topics,
                             raw_data=matrix,
                             path=self.path + "lda/kmeans/",
                             prefix=self.prefix,
                             soft_clustering=False)

        return cluster_assignments, topics

    def __applyKMeans__(self, raw_data, vocab, soft_clustering):
        kmeans = KMeans(n_clusters=self.n_topics, random_state=0).fit(soft_clustering)
        cluster_assignments = kmeans.labels_
        clusters_unique = np.unique(cluster_assignments)  # sorted unique elements

        # Derive Topics (with top-n words per cluster)
        topics = {}
        # for i, row in enumerate(Vt):
        #    topics[i] = vocab[np.argsort(row)[::-1][:top_n]]
        for c_idx, cluster in enumerate(clusters_unique):
            term_weights = np.asfarray(csr_matrix(raw_data)[cluster_assignments == cluster, :].sum(axis=0)).flatten()
            top = term_weights.argsort()[::-1][:self.top_n]
            top = top[term_weights[top] > 0]
            topics[c_idx] = {"terms": vocab[top],
                             "weights": term_weights[top]}

        return cluster_assignments, topics


    def __postprocess__(self, clusters, topics, raw_data, path, prefix, soft_clustering=True, storeToDB=False, uris=None):
        self.__calc_metrics__(topics=topics,
                              cluster_assignments=clusters,
                              raw_data=raw_data,
                              soft_clustering=soft_clustering)

        out.filterTopics(topics=topics,
                         soft_clustering=soft_clustering)

        out.print_clusters(cluster_assignments=clusters,
                           topics=topics,
                           soft_clustering=soft_clustering)
        print("Mean Silhouette Score: " + str(out.calcMeanSilhouetteScore(topics)))
        # if hardclustering:
        #     print("KMeans: Calculating ", n_topics, " clusters (topics)...")
        #     cluster_assignments, topics = self.__applyKMeans__(n_topics=n_topics,
        #                                               raw_data=matrix,
        #                                               vocab=vocab,
        #                                               top_n=top_n,
        #                                               soft_clustering=U)
        #     print("KMeans: ", len(cluster_assignments), " cluster assignments")

        out.create_wordclouds(cluster_assignments=clusters,
                              topics=topics,
                              files_path=path,
                              prefix=prefix,
                              clear_path=True,
                              soft_clustering=soft_clustering)
        if storeToDB:
            out.storeClustersToDB(cluster_assignments=clusters,
                                  topics=topics,
                                  source_uris=uris,
                                  soft_clustering=soft_clustering)
        return


    def __calc_metrics__(self, topics, cluster_assignments, raw_data, soft_clustering=True):
        print("Calculate KPIs...")
        total_count = cluster_assignments.shape[0]

        if soft_clustering:
            sums = np.sum(cluster_assignments, axis=0)
            counts = np.count_nonzero(cluster_assignments, axis=0)
            hard_cluster = np.argsort(cluster_assignments, axis=1)[:, -1]  # just the last column (Ascending sorted!)
            # inter-cluster-sim - intra-cluster-sim / max of both
            silhouette_scores = silhouette_samples(raw_data, hard_cluster, metric='cosine')
        else:
            # inter-cluster-sim - intra-cluster-sim / max of both
            silhouette_scores = silhouette_samples(raw_data, cluster_assignments, metric='cosine')

        for c_idx, topic in topics.items():
            # Calc. KPIs

            if soft_clustering:
                count = counts[c_idx]
                avg_weight = sums[c_idx] / counts[c_idx]
                article_ratio = count / total_count
                mask_nonzero = cluster_assignments[:, c_idx] > 0
                std = np.std(cluster_assignments[mask_nonzero, c_idx])
                median = np.median(cluster_assignments[mask_nonzero, c_idx])
                top_ratio = len(np.where(hard_cluster == c_idx)[0]) / counts[c_idx]
                silhouette_score = np.mean(silhouette_scores[hard_cluster == c_idx])

                # Store KPIs
                topic.update({"count": count,
                              "avg_weight": avg_weight,
                              "median_weight": median,
                              "std_weight": std,
                              "article_ratio": article_ratio,
                              "top_ratio": top_ratio,
                              "silhouette_score": silhouette_score})
            else:
                count = len(cluster_assignments[cluster_assignments == c_idx])
                article_ratio = count / total_count
                silhouette_score = np.mean(silhouette_scores[cluster_assignments == c_idx])
                # Store KPIs
                topic.update({"count": count,
                              "article_ratio": article_ratio,
                              "silhouette_score": silhouette_score})

        return  # topics  #inplace update!


    def __removeInvalid__(self,topics, cluster_assignments):
        delete_idx = []
        for idx, topic in topics.items():
            if not topic["keep"]:
                delete_idx.append(idx)
        return np.delete(cluster_assignments,delete_idx,axis=1)




