from pyjarowinkler import distance
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk, string
#from scipy.spatial import distance
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn.metrics import jaccard_similarity_score
from sklearn.feature_extraction.text import CountVectorizer
from math import*
from decimal import Decimal
import re



def cek_typo(kunci_jawaban, jawaban, toleransi=0.95):
  jawaban =  jawaban.lower()
  kunci_jawaban_split = kunci_jawaban.split()
  jawaban_split = jawaban.split()
  n_jawaban = jawaban_split
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

def df(text1, text2, fitur):
    #vectorizer = TfidfVectorizer(vocabulary=fitur)
    vectorizer_df = CountVectorizer(vocabulary=fitur)

    tfidf = vectorizer_df.fit_transform([text1, text2])
    #df = vectorizer_df.fit_transform([text1, text2])
    # return ((tfidf * tfidf.T).A)[0,1]
    #tfidf_ = np.matrix(tfidf.A * df.A)
    return tfidf

def tf_idf(text1, text2, fitur):
    vectorizer = TfidfVectorizer(vocabulary=fitur)
    tfidf = vectorizer.fit_transform([text1, text2])
    return tfidf

#def cosine_sim_df(text1, text2, fitur):
##cosine_similarity
def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)
def cosine_similarity(x, y):
    if sum(y) == 0:
        return 0.0
    else:
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = square_rooted(x) * square_rooted(y)
        hasil = round(numerator / float(denominator), 3)
        return hasil
##jaccard_similarity
#def nth_root(value, n_root):
    #root_value = 1 / float(n_root)
    #return round(Decimal(value) ** Decimal(root_value), 3)

def total (summ):
    jumlah = 0
    for i in summ:
        jumlah += i
    return jumlah

#t = nawa.df(kunci_jawaban, jawaban_ct, fitur)
def jaccard_baru(x, y):
    numerator = [a * b for a, b in zip(x, y)]
    x_ = sum(x**2)#x**2
    y_ = sum(y**2)#
    denominator =  (x_ + y_ )- sum(numerator)
    return np.array(sum(numerator)) / np.array(denominator)

def dice_similarity_(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:

        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def dice_similarity(x, y):
    numerator = [a * b for a, b in zip(x, y)]
    x_ = sum(x**2)
    y_ = sum(y**2)
    denominator =  x_ + y_
    return np.array(2*(sum(numerator))) / np.array(denominator)
    #return np.array((sum(numerator))) / np.array(denominator)
def ubah_simbol(teks):
    return teks.replace(".", "").replace("}", "").replace("{", "").replace("(", "").replace(")", "").replace("-", "").replace(":", " ")

def pisahKata(kunci_jawaban, jawaban):
    d_index = []
    b_index = []
    
    for i in kunci_jawaban:
        index_replace = [(m.start(0)) for m in re.finditer(i,jawaban)]
        d_index += index_replace
    
        index_replace = [(m.end(0)) for m in re.finditer(i,jawaban)]
        b_index += index_replace
    jawaban_list = [x for x in jawaban]
    for d, b in zip(d_index, b_index):
        #print(d,b)
        jawaban_list[d]= " "+jawaban_list[d]
        jawaban_list[b-1]=jawaban_list[b-1]+" "

    jawaban_list = re.sub(r"\s+", " ","".join(jawaban_list).rstrip().strip().lstrip())
    return jawaban_list

def cek_negasi(kata_negasi, kata_dicari):
    if type(kata_negasi) != list:
        kata_negasi = [kata_negasi]  
    n_index = []
    for i in kata_negasi:
        index_replace = [(m.end(0)) for m in re.finditer(i,kata_dicari)]
        n_index += index_replace
        #print(n_index)
    for rep in n_index:
        if rep != len(kata_dicari):
            huruf = [x for x in kata_dicari]
            huruf[rep] = "_"
            kata_dicari = "".join(huruf)
    return kata_dicari
