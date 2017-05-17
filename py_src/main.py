from datahandler import DataHandler
import decompositions as dc
import output as out
from clusterhandler import ClusterHandler

def main():
    n_topics = 40
    offset = 0
    top_n = 20 #words per topic
    prefix = "t40_12gram_"
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
    cluster_assignments, topics = ch.calc_nmf(matrix=tfidf,
                                              vocab=tfidf_vocab,
                                              providers=dh.get_providers(),
                                              hardclustering=False)

    out.storeClustersToDB(cluster_assignments=cluster_assignments,
                          topics=topics,
                          source_uris=dh.get_uris(),
                          soft_clustering=True)

    # for max_iter in range(10,101,50):
    #     ch.calc_lda(matrix=tf, vocab=tf_vocab,max_iter=max_iter)




if __name__ == '__main__':
    main()


