from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse.linalg import svds

from dbhandler import DatabaseHandler

print("Querying data...")
handler = DatabaseHandler("ec2-52-57-13-180.eu-central-1.compute.amazonaws.com", "webmining", "asN5O$YVZch-$vyFEN^*", "webmining")
result = handler.execute(
    """SELECT source_uri as 'uri', text 
    FROM NewsArticles    
    """)


n = len(result)
uris = np.empty(n,dtype=np.dtype(('U', 255)))
texts = np.empty(n,dtype=np.dtype(('U', 10000)))

for i, row in enumerate(result):
    uris[i] = row["uri"]
    texts[i] = row["text"]

#Sparse Term-Doc matix (TF-IDF)
print("Calculation TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   #max_features=n_features,
                                   stop_words='english',
                                   sublinear_tf=True)
tfidf = tfidf_vectorizer.fit_transform(texts)
print("Finished TF-IDF: ", tfidf.shape)

#SVD
k = 20
print("Calculating SVD with k=", k)

U, s, Vt = svds(tfidf,k=k)
vocab = np.array(tfidf_vectorizer.get_feature_names())

print(vocab)
topn = 20
print("Top ",topn, " words per topic")
for i, row in enumerate(U.transpose()):
    print("Topic ", i, ":")
    print(vocab[np.argsort(row)[::-1][:topn]])

