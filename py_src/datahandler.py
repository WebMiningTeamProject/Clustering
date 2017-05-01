from dbhandler import DatabaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

class DataHandler:
    def __init__(self, use_cache, files_path="files/"):
        #Init Data
        self.uris, self.texts = self.__load_data(use_cache=use_cache, files_path=files_path)

        #Vectorize
        self.tfidf, self.tfidf_vocab = self.__calc_tfidf(self.texts)
        self.tf, self.tf_vocab = self.__calc_tf(self.texts)

    def get_uris(self):
        return self.uris

    def get_texts(self):
        return self.texts

    def get_tfidf(self):
        return self.tfidf, self.tfidf_vocab

    def get_tf(self):
        return self.tf, self.tf_vocab

    def __load_data(self, use_cache=True, files_path="files/"):
        if (use_cache):
            try:
                uris = np.load(files_path + "uris.npy")
                texts = np.load(files_path + "texts.npy")
                print("Cached data loaded from ", files_path)
            except:
                use_cache = False
                print("No cached data found in ", files_path)

        if (not use_cache):
            print("Data: Querying ...")
            handler = DatabaseHandler()
            result = handler.execute(
                """SELECT source_uri as 'uri', bow as 'text' 
                FROM NewsArticlesBOW    
                """)

            n = len(result)
            uris = np.empty(n, dtype=np.dtype(('U', 255)))
            texts = np.empty(n, dtype=np.dtype(('U', 10000)))

            for i, row in enumerate(result):
                uris[i] = row["uri"]
                texts[i] = row["text"]
            print("Data: Retrieved ", len(texts), " texts")

            try:
                np.save(files_path + "uris.npy", uris)
                np.save(files_path + "texts.npy", texts)
                print("Cached data in ", files_path)
            except:
                print("Could not cache data in ", files_path)

        return uris, texts

    def __calc_tfidf(self, texts):
        # Sparse Doc-Term(!) matix (TF-IDF)
        print("TF-IDF: Calculation...")
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           # max_features=n_features,
                                           # stop_words='english',
                                           sublinear_tf=True)
        tfidf = tfidf_vectorizer.fit_transform(texts)
        vocab = np.array(tfidf_vectorizer.get_feature_names())
        print("TF-IDF: Finished, shape: ", tfidf.shape)
        # print("TF-IDF: Vocab shape: ", vocab.shape)
        return tfidf, vocab

    def __calc_tf(self, texts):
        # Sparse Doc-Term(!) matix (TF)
        print("TF: Calculation...")
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
        tf = tf_vectorizer.fit_transform(texts)
        vocab = np.array(tf_vectorizer.get_feature_names())
        print("TF: Finished, shape: ", tf.shape)
        return tf, vocab
