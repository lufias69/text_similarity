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




def pisahKata_old(dicari, jawaban):
    dicari_split = dicari.split()
    jawaban_split = jawaban.split()
    for h in dicari_split:
        for ct, i in enumerate(jawaban_split) :
            p_dicari = len(h)
            index_depan = 0
            if re.search(h, i):
                index_depan = re.search(h, i).start()
                index_akhir = re.search(h, i).end()
            if len(i) >= p_dicari:
                if re.search(h, i):
                    i = [x for x in i]
                    i.insert(index_akhir, " ")
                    if index_depan != 0:
                        i.insert(index_depan, " ")
                    i = "".join(i)
                    jawaban_split[ct] = i
    return " ".join(jawaban_split)

def pisahKata_old(kunci_jawaban, jawaban):
    kunci_jawaban_split = kunci_jawaban.split()
    jawaban_split = jawaban.split()
    kate = []
    for i in kunci_jawaban_split:
        for j in jawaban_split:
            if i != j:
                if re.search(i, j):
                    kate.append(i)

    #jawaban_split = " ".join(list(set(jawaban_split+kate)))
    return " ".join(list(set(jawaban_split+kate)))

def pisahKata__old(dicari, jawaban):
    dicari_split = dicari.split()
    jawaban_split = jawaban.split()
    for h in dicari_split:
        for ct, i in enumerate(jawaban_split) :
            p_dicari = len(h)
            if len(i) >= p_dicari:
                a = []
                for j in range(p_dicari):
                    a.append(i[j])
                a_j = "".join(a)
                if a_j == h:
                    #print(i)
                    i = [x for x in i]
                    i.insert(p_dicari, " ")
                    i = "".join(i)
                    jawaban_split[ct] = i
    return " ".join(jawaban_split)

def pisahKata(kunci_jawaban, jawaban):
    kunci_jawaban_split = kunci_jawaban.split()
    jawaban_split       = jawaban.split()
    n_jawaban = []
    for kj in kunci_jawaban_split:
        for key, j in enumerate(jawaban_split):
            if re.search(kj, j):
                start = re.search(kj, j).start()
                end   = re.search(kj, j).end()
                if start != 0 and end != len(j):
                    #print(kj)
                    j_ = [x for x in j]
                    if j_[start-1] != " ":
                        
                        j_[start] = " "+j_[start]
                    if j_[end] != " ":
                        j_[end] = " "+j_[end-1]
                        #print(j_)
                        #print(end)
                        j = "".join(j_)
                        jawaban_split[key]=j
                        #print(j)
                
                elif start != 0 and end == len(j):
                    #print(kj)
                    j_ = [x for x in j]
                    if j_[start-1] != " ":
                        
                        j_[start] = " "+j_[start]
                    
                    #j_[end] = " "+j_[end-1]
                    #print(j_)
                    #print(end)
                    j = "".join(j_)
                    jawaban_split[key]=j
                    #print(j)
                    #pass
                elif start == 0 and end != len(j):
                    #print(kj)
                    j_ = [x for x in j]
                    #if j_[start-1] != " ":
                        #print("ini")
                        #j_[start] = " "+j_[start]
                    if j_[end] != " ":
                        j_[end] = " "+ j_[end]
                        #print(j_)
                        #print(end)
                        j = "".join(j_)
                        jawaban_split[key]=j
                        #print(j)
                    
                else:
                    #print("ini else", j)
                    pass
            #n_jawaban.append(j)
    return " ".join(jawaban_split)

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
