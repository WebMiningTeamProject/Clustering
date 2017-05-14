from datahandler import DataHandler
import decompositions as dc
import output as out
from clusterhandler import ClusterHandler

def main():
    n_topics = 21
    offset = 22
    top_n = 20 #words per topic
    prefix = "t21o22_12gram_"
    path = "files/wordclouds/"
    ngram_range = (1, 2)

    dh = DataHandler(use_cache=True, ngram_range=ngram_range)
    tfidf, tfidf_vocab = dh.get_tfidf()
    tf, tf_vocab = dh.get_tf()

    # for n_topics in range(20, 27, 1):
    #     for offset in range (n_topics-2,n_topics+3,1):
    ch = ClusterHandler(n_topics=n_topics,
                        top_n=top_n,  # words per topic
                        soft_offset=offset,
                        prefix=prefix,
                        path=path,
    )

    #ch.calc_svd(matrix=tfidf, vocab=tfidf_vocab)
    ch.calc_nmf(matrix=tfidf, vocab=tfidf_vocab)

        # for max_iter in range(10,101,50):
        #     ch.calc_lda(matrix=tf, vocab=tf_vocab,max_iter=max_iter)


    # tfidf, tfidf_vocab = dh.get_tfidf()
    #
    # #Do SVD
    # clusters, topics = dc.calc_svd(matrix=tfidf,
    #                                vocab=tfidf_vocab,
    #                                svd_k=n_topics + 5,
    #                                n_topics=n_topics,
    #                                top_n=top_n)
    # out.filterTopics(topics)
    # out.print_clusters(clusters, topics)
    # out.create_wordclouds(clusters, topics,files_path=path + "svd/",prefix=prefix)
    #
    # #Do NMF
    # clusters, topics = dc.calc_nmf(matrix=tfidf,
    #                                vocab=tfidf_vocab,
    #                                n_topics=n_topics,
    #                                top_n=top_n)
    # postprocess(clusters=clusters,
    #             topics=topics,
    #             path=path + "nmf/",
    #             prefix=prefix,
    #             hardclustering=False,
    #             storeToDB=False,
    #             uris=dh.get_uris()
    #             )
    #
    # #Do LDA:
    # tf, tf_vocab = dh.get_tf()
    # clusters, topics = dc.calc_lda(matrix=tf,
    #                                vocab=tf_vocab,
    #                                n_topics=n_topics,
    #                                top_n=top_n)
    # out.filterTopics(topics)
    # out.print_clusters(clusters, topics)
    # out.create_wordclouds(clusters, topics, files_path=path + "lda/",prefix=prefix)


# def postprocess(clusters, topics, path, prefix, hardclustering=False, storeToDB=False, uris=None):
#     out.filterTopics(topics)
#     out.print_clusters(clusters, topics)
#     if hardclustering:
#         print("KMeans: Calculating ", n_topics, " clusters (topics)...")
#         cluster_assignments, topics = applyKMeans(n_topics=n_topics,
#                                                   raw_data=matrix,
#                                                   vocab=vocab,
#                                                   top_n=top_n,
#                                                   soft_clustering=U)
#         print("KMeans: ", len(cluster_assignments), " cluster assignments")
#
#     out.create_wordclouds(clusters, topics, files_path=path, prefix=prefix, clear_path=True)
#     if storeToDB:
#         out.storeClustersToDB(cluster_assignments=clusters, topics=topics, source_uris=uris)
#
#
# def applyKMeans(n_topics,raw_data, vocab, top_n, soft_clustering):
#     kmeans = KMeans(n_clusters=n_topics, random_state=0).fit(soft_clustering)
#     cluster_assignments = kmeans.labels_
#     clusters_unique = np.unique(cluster_assignments)  # sorted unique elements
#
#     # Derive Topics (with top-n words per cluster)
#     topics = {}
#     # for i, row in enumerate(Vt):
#     #    topics[i] = vocab[np.argsort(row)[::-1][:top_n]]
#     for c_idx, cluster in enumerate(clusters_unique):
#         term_weights = np.asfarray(csr_matrix(raw_data)[cluster_assignments == cluster, :].sum(axis=0)).flatten()
#         top = term_weights.argsort()[::-1][:top_n]
#         top = top[term_weights[top] > 0]
#         topics[c_idx] = {"terms": vocab[top],
#                          "weights": term_weights[top]}
#
#     return cluster_assignments, topics



if __name__ == '__main__':
    main()


