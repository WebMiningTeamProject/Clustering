from dbhandler import DatabaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from scipy.sparse import load_npz, save_npz

class DataHandler:
    def __init__(self, use_cache, files_path="files/", ngram_range=(1, 1)):
        #Init Data
        self.valid = self.__load_data(use_cache=use_cache, files_path=files_path, ngram_range=ngram_range)

    def get_uris(self):
        return self.uris

    def get_providers(self):
        return self.providers

    def get_texts(self):
        return self.texts

    def get_tfidf(self):
        return self.tfidf, self.tfidf_vocab

    def get_tf(self):
        return self.tf, self.tf_vocab

    def __load_data(self, use_cache=True, files_path="files/", ngram_range=(1, 1)):
        valid = False
        if (use_cache):
            try:
                self.providers = np.load(files_path + "providers.npy")
                self.uris = np.load(files_path + "uris.npy")
                self.texts = np.load(files_path + "texts.npy")
                self.tfidf = load_npz(files_path + "tfidf.npz")
                self.tfidf_vocab = np.load(files_path + "tfidf_vocab.npy")
                self.tf = load_npz(files_path + "tf.npz")
                self.tf_vocab = np.load(files_path + "tf_vocab.npy")

                valid = True
                print("Cached data loaded from ", files_path)
            except:
                use_cache = False
                print("No cached data found in ", files_path)

        if (not use_cache):
            print("Data: Querying ...")
            handler = DatabaseHandler()
            result = handler.execute(
                """SELECT b.source_uri as 'uri', b.bow as 'text', p.root_name as 'provider'
                FROM NewsArticlesBOW as b 
                INNER JOIN NewsArticles as a ON b.source_uri = a.source_uri
	            INNER JOIN NewsProviderComplete as p ON a.news_provider = p.name
                """)

            n = len(result)
            self.uris = np.empty(n, dtype=np.dtype(('U', 255)))
            self.texts = np.empty(n, dtype=np.dtype(('U', 10000)))
            self.providers = np.empty(n, dtype=np.dtype(('U', 255)))

            for i, row in enumerate(result):
                self.uris[i] = row["uri"]
                self.texts[i] = row["text"]
                self.providers[i] = row["provider"]
            print("Data: Retrieved ", len(self.texts), " texts")

            # Vectorize
            self.tfidf, self.tfidf_vocab = self.__calc_tfidf(self.texts, ngram_range=ngram_range)
            self.tf, self.tf_vocab = self.__calc_tf(self.texts, ngram_range=ngram_range)

            valid = True

            try:
                np.save(files_path + "providers.npy", self.providers)
                np.save(files_path + "uris.npy", self.uris)
                np.save(files_path + "texts.npy", self.texts)
                np.save(files_path + "tfidf_vocab.npy", self.tfidf_vocab)
                np.save(files_path + "tf_vocab.npy", self.tf_vocab)

                save_npz(files_path + "tfidf.npz", self.tfidf)
                save_npz(files_path + "tf.npz", self.tf)

                print("Cached data in ", files_path)
            except:
                print("Could not cache data in ", files_path)

        return valid

    def __calc_tfidf(self, texts, ngram_range=(1, 1)):
        # Sparse Doc-Term(!) matix (TF-IDF)
        print("TF-IDF: Calculation...")
        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=10,
                                           # max_features=n_features,
                                           # stop_words='english',
                                           sublinear_tf=True,
                                           ngram_range=ngram_range)
        tfidf = tfidf_vectorizer.fit_transform(texts)
        vocab = np.array(tfidf_vectorizer.get_feature_names())
        print("TF-IDF: Finished, shape: ", tfidf.shape)
        # print("TF-IDF: Vocab shape: ", vocab.shape)
        return tfidf, vocab

    def __calc_tf(self, texts, ngram_range=(1, 1)):
        # Sparse Doc-Term(!) matix (TF)
        print("TF: Calculation...")
        tf_vectorizer = CountVectorizer(max_df=0.8,
                                        min_df=10,
                                        ngram_range=ngram_range)
        tf = tf_vectorizer.fit_transform(texts)
        vocab = np.array(tf_vectorizer.get_feature_names())
        print("TF: Finished, shape: ", tf.shape)
        return tf, vocab
