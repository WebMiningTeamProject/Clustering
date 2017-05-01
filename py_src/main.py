from datahandler import DataHandler
import decompositions as dc
import output as out

def main():
    n_topics = 15
    top_n = 20 #words per topic

    # uris, texts = get_data(use_cache=True)
    # tfidf, vocab = get_tfidf(texts)

    dh = DataHandler(use_cache=True)

    tfidf, tfidft_vocab = dh.get_tfidf()

    #Do SVD
    clusters, topics = dc.calc_svd(matrix=tfidf,
                                vocab=tfidft_vocab,
                                svd_k=20,
                                n_topics=n_topics,
                                top_n=top_n)
    out.print_clusters(clusters, topics)
    out.create_wordclouds(clusters, topics,files_path="files/wordclouds/svd/")

    #Do NMF
    clusters, topics = dc.calc_nmf(matrix=tfidf,
                                vocab=tfidft_vocab,
                                n_topics=n_topics,
                                top_n=top_n)
    out.print_clusters(clusters,topics)
    out.create_wordclouds(clusters, topics, files_path="files/wordclouds/nmf/")

    #Do LDA:
    tf, tf_vocab = dh.get_tf()
    clusters, topics = dc.calc_lda(matrix=tf,
                                vocab=tf_vocab,
                                n_topics=n_topics,
                                top_n=top_n)
    out.print_clusters(clusters, topics)
    out.create_wordclouds(clusters, topics, files_path="files/wordclouds/lda/")



if __name__ == '__main__':
    main()


