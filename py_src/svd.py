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

#Sparse Doc-Term(!) matix (TF-IDF)
print("Calculation TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   #max_features=n_features,
                                   stop_words='english',
                                   sublinear_tf=True)
tfidf = tfidf_vectorizer.fit_transform(texts)
print("Finished TF-IDF: ", tfidf.shape)

#SVD
k = 100
print("Calculating SVD with k=", k)

U, s, Vt = svds(tfidf,k=k)
vocab = np.array(tfidf_vectorizer.get_feature_names())
print("S: ", s[::-1])
#print("Vocab: ",vocab.shape)
topn = 20
print("Top ",topn, " words for first 10 topics")
for i, row in enumerate(Vt[k-10:,:]):
    print("Topic ", i, ":")
    #print("Row: ", row.shape)
    print(vocab[np.argsort(row)[::-1][:topn]])

