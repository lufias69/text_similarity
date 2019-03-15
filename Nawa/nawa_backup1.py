from pyjarowinkler import distance
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk, string
#from scipy.spatial import distance
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from sklearn.feature_extraction.text import CountVectorizer
def cek_typo(kunci_jawaban, jawaban, toleransi=0.95):

  kunci_jawaban_split = kunci_jawaban.split()
  jawaban_split = jawaban.split()
  for i in range(len(jawaban_split)):
    w_1 = []
    n_jawaban = jawaban_split
    kunci_jawaban_ = []
    for j in kunci_jawaban_split:
      w_1.append(distance.get_jaro_distance(jawaban_split[i], j, winkler=True, scaling=0.1))
      kunci_jawaban_.append(j)
    if max(w_1) != 1.0 and max(w_1) > toleransi:
      index = w_1.index(max(w_1))
      n_jawaban[i]= kunci_jawaban_[index]
     
      
  return " ".join(n_jawaban)


def tf_idf_df(text1, text2, fitur):
    vectorizer = TfidfVectorizer(vocabulary=fitur)
    vectorizer_df = CountVectorizer(vocabulary=fitur)

    tfidf = vectorizer.fit_transform([text1, text2])
    df = vectorizer_df.fit_transform([text1, text2])
    # return ((tfidf * tfidf.T).A)[0,1]
    tfidf_ = np.matrix(tfidf.A * df.A)
    return tfidf_

def cosine_sim(text1, text2, fitur):
    vectorizer = TfidfVectorizer(vocabulary = fitur)
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

#def cosine_sim_df(text1, text2, fitur):
def cosine_sim_df(tff):
    #vectorizer = TfidfVectorizer(vocabulary=fitur)
    #vectorizer_df = CountVectorizer(vocabulary=fitur)

    #tfidf = vectorizer.fit_transform([text1, text2])
    #df = vectorizer_df.fit_transform([text1, text2])
    # return ((tfidf * tfidf.T).A)[0,1]
    #tfidf_ = np.matrix(text1.A * df.A)
    #return ((tfidf_ * tfidf_.T).A)[0,1]
    return ((tff * tff.T).A)[0, 1]

def jaccard_sim(text1, text2, fitur):
    vectorizer = TfidfVectorizer(vocabulary = fitur)
    tfidf = vectorizer.fit_transform([text1, text2])
    tfidf = tfidf.toarray()
    str1 = tfidf[0]
    str2 = tfidf[1]
    intersection = len(list(set(str1).intersection(str2)))
    union = (len(str1) + len(str2)) - intersection
    return float(intersection / union)

def jaccard_sim_sklearn(text1, text2, fitur):
    vectorizer = TfidfVectorizer(vocabulary = fitur)
    tfidf = vectorizer.fit_transform([text1, text2])
    tfidf = tfidf.toarray()
    y_pred = tfidf[0].tolist()
    y_true = tfidf[1].tolist()
    return jaccard_similarity_score(y_true, y_pred)
    #return jaccard_similarity_score(y_true, y_pred, normalize=False)
def cosine_sim_tf_idf_df(text1, text2, fitur):
    vectorizer = TfidfVectorizer(vocabulary = fitur)
    vectorizer_df = CountVectorizer(vocabulary = fitur)

    tfidf = vectorizer.fit_transform([text1, text2])
    df = vectorizer_df.fit_transform([text1, text2])

    return ((tfidf * tfidf.T).A)[0,1]

