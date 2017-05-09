from datahandler import DataHandler
import decompositions as dc
import output as out

def main():
    n_topics = 30
    top_n = 20 #words per topic
    prefix = "t30_12gram_"
    path = "files/wordclouds/"
    ngram_range = (1, 2)

    dh = DataHandler(use_cache=True,ngram_range=ngram_range)

    tfidf, tfidf_vocab = dh.get_tfidf()

    #Do SVD
    # clusters, topics = dc.calc_svd(matrix=tfidf,
    #                                vocab=tfidf_vocab,
    #                                svd_k=n_topics + 5,
    #                                n_topics=n_topics,
    #                                top_n=top_n)
    # out.print_clusters(clusters, topics)
    # out.create_wordclouds(clusters, topics,files_path=path + "svd/",prefix=prefix)

    #Do NMF
    clusters, topics = dc.calc_nmf(matrix=tfidf,
                                   vocab=tfidf_vocab,
                                   n_topics=n_topics,
                                   top_n=top_n)
    out.filterTopics(topics)
    out.print_clusters(clusters,topics)
    out.create_wordclouds(clusters, topics, files_path=path + "nmf/",prefix=prefix,clear_path=True)
    out.storeClustersToDB(cluster_assignments=clusters, topics=topics, source_uris=dh.get_uris())

    #Do LDA:
    # tf, tf_vocab = dh.get_tf()
    # clusters, topics = dc.calc_lda(matrix=tf,
    #                                vocab=tf_vocab,
    #                                n_topics=n_topics,
    #                                top_n=top_n)
    # out.print_clusters(clusters, topics)
    # out.create_wordclouds(clusters, topics, files_path=path + "lda/",prefix=prefix)


if __name__ == '__main__':
    main()


