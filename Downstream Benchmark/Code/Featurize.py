import pickle
import pandas as pd
import numpy as np
import os
from pandas.api.types import is_numeric_dtype
from collections import Counter,defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from dirty_cat import SimilarityEncoder

def URLProcessor(x):
    y = re.findall(r"[\w']+", x)
    z = ' '.join(y)
    return z

def Featurize(dataDownstream_train,dataDownstream_test,dataDownstream_heldtest,attribute_names,y_cur):

    all_cols_train,numeric_cols_train,categ_cols_train,ngram_cols_train,url_cols_train = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    all_cols_test,numeric_cols_test,categ_cols_test,ngram_cols_test,url_cols_test = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    all_cols_heldtest,numeric_cols_heldtest,categ_cols_heldtest,ngram_cols_heldtest,url_cols_heldtest = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    
    vectorizer = CountVectorizer(ngram_range=(2,2),analyzer='char')
    vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
    vectorizerWord = CountVectorizer(ngram_range=(2,2))
    enc = OneHotEncoder(handle_unknown='ignore')
    
    for i in range(len(y_cur)):
        curcol = attribute_names[i]
        curdf_train = dataDownstream_train[curcol]
        curdf_test = dataDownstream_test[curcol]
        curdf_heldtest = dataDownstream_heldtest[curcol]
#         print(curcol)
        
        if y_cur[i] == 0:
            numeric_cols_train = pd.concat([numeric_cols_train,curdf_train],axis=1)
            numeric_cols_test = pd.concat([numeric_cols_test,curdf_test],axis=1)
            numeric_cols_heldtest = pd.concat([numeric_cols_heldtest,curdf_heldtest],axis=1)            
        if y_cur[i] in [1,10]:
            curdf_train = curdf_train.astype(str)
            curdf_test = curdf_test.astype(str)
            curdf_heldtest = curdf_heldtest.astype(str)            
            
            tempdf_train = pd.DataFrame(enc.fit_transform(curdf_train.to_frame()).toarray())
            tempdf_test = pd.DataFrame(enc.transform(curdf_test.to_frame()).toarray())
            tempdf_heldtest = pd.DataFrame(enc.transform(curdf_heldtest.to_frame()).toarray())
            
            categ_cols_train = pd.concat([categ_cols_train,tempdf_train],axis=1)
            categ_cols_test = pd.concat([categ_cols_test,tempdf_test],axis=1)
            categ_cols_heldtest = pd.concat([categ_cols_heldtest,tempdf_heldtest],axis=1)
        if y_cur[i] == 3:
            arr_train = curdf_train.astype(str).values
            X_train = vec.fit_transform(arr_train)
            tempdf_train = pd.DataFrame(X_train.toarray())
            
            arr_test = curdf_test.astype(str).values
            X_test = vec.transform(arr_test)
            tempdf_test = pd.DataFrame(X_test.toarray())
            
            arr_heldtest = curdf_heldtest.astype(str).values
            X_heldtest = vec.transform(arr_heldtest)
            tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())            
            
            ngram_cols_train = pd.concat([ngram_cols_train,tempdf_train], axis=1, sort=False)
            ngram_cols_test = pd.concat([ngram_cols_test,tempdf_test], axis=1, sort=False)
            ngram_cols_heldtest = pd.concat([ngram_cols_heldtest,tempdf_heldtest], axis=1, sort=False)            
            
        if y_cur[i] == 4:
            temp_train = curdf_train.apply(lambda x: URLProcessor(x))
            arr_train = temp_train.astype(str).values
            X_train = vectorizerWord.fit_transform(arr_train)
            tempdf_train = pd.DataFrame(X_train.toarray())

            temp_test = curdf_test.apply(lambda x: URLProcessor(x))
            arr_test = temp_test.astype(str).values
            X_test = vectorizerWord.fit_transform(arr_test)
            tempdf_test = pd.DataFrame(X_test.toarray())
            
            temp_heldtest = curdf_heldtest.apply(lambda x: URLProcessor(x))
            arr_heldtest = temp_heldtest.astype(str).values
            X_heldtest = vectorizerWord.fit_transform(arr_heldtest)
            tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())            
            
            url_cols_train = pd.concat([url_cols_train,tempdf_train], axis=1, sort=False)
            url_cols_test = pd.concat([url_cols_test,tempdf_test], axis=1, sort=False)
            url_cols_heldtest = pd.concat([url_cols_heldtest,tempdf_heldtest], axis=1, sort=False)            
            
        if y_cur[i] in [2,3,5,6,8]:
            arr_train = curdf_train.astype(str).values
            X_train = vectorizer.fit_transform(arr_train)
            tempdf_train = pd.DataFrame(X_train.toarray())
            
            arr_test = curdf_test.astype(str).values
            X_test = vectorizer.transform(arr_test)
            tempdf_test = pd.DataFrame(X_test.toarray())
            
            arr_heldtest = curdf_heldtest.astype(str).values
            X_heldtest = vectorizer.transform(arr_heldtest)
            tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())
            
            ngram_cols_train = pd.concat([ngram_cols_train,tempdf_train], axis=1, sort=False)
            ngram_cols_test = pd.concat([ngram_cols_test,tempdf_test], axis=1, sort=False)
            ngram_cols_heldtest = pd.concat([ngram_cols_heldtest,tempdf_heldtest], axis=1, sort=False)            
            
    all_cols_train = pd.concat([all_cols_train,numeric_cols_train,categ_cols_train,ngram_cols_train], axis=1, sort=False)
    all_cols_test = pd.concat([all_cols_test,numeric_cols_test,categ_cols_test,ngram_cols_test], axis=1, sort=False)
    all_cols_heldtest = pd.concat([all_cols_heldtest,numeric_cols_heldtest,categ_cols_heldtest,ngram_cols_heldtest], axis=1, sort=False)
    
    return all_cols_train,all_cols_test,all_cols_heldtest


def FeaturizeSimiarity(dataDownstream_train,dataDownstream_test,dataDownstream_heldtest,attribute_names,y_cur):

    all_cols_train,numeric_cols_train,categ_cols_train,ngram_cols_train,url_cols_train = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    all_cols_test,numeric_cols_test,categ_cols_test,ngram_cols_test,url_cols_test = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    all_cols_heldtest,numeric_cols_heldtest,categ_cols_heldtest,ngram_cols_heldtest,url_cols_heldtest = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    
    vectorizer = CountVectorizer(ngram_range=(2,2),analyzer='char')
    vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
    vectorizerWord = CountVectorizer(ngram_range=(2,2))
    enc = OneHotEncoder(handle_unknown='ignore')
    similarity_encoder = SimilarityEncoder(similarity='ngram')
    
    for i in range(len(y_cur)):
        curcol = attribute_names[i]
        curdf_train = dataDownstream_train[curcol]
        curdf_test = dataDownstream_test[curcol]
        curdf_heldtest = dataDownstream_heldtest[curcol]
        
        # print(curdf_heldtest)

        if y_cur[i] == 0:
            numeric_cols_train = pd.concat([numeric_cols_train,curdf_train],axis=1)
            numeric_cols_test = pd.concat([numeric_cols_test,curdf_test],axis=1)
            numeric_cols_heldtest = pd.concat([numeric_cols_heldtest,curdf_heldtest],axis=1)
        if y_cur[i] in [1]:
            tempdf_train = pd.DataFrame(enc.fit_transform(curdf_train.to_frame()).toarray())
            tempdf_test = pd.DataFrame(enc.transform(curdf_test.to_frame()).toarray())
            tempdf_heldtest = pd.DataFrame(enc.transform(curdf_heldtest.to_frame()).toarray())
            
            categ_cols_train = pd.concat([categ_cols_train,tempdf_train],axis=1)
            categ_cols_test = pd.concat([categ_cols_test,tempdf_test],axis=1)
            categ_cols_heldtest = pd.concat([categ_cols_heldtest,tempdf_heldtest],axis=1)            
        if y_cur[i] in [10]:
            tempdf_train = pd.DataFrame(similarity_encoder.fit_transform(curdf_train.values.reshape(-1, 1)))
            tempdf_test = pd.DataFrame(similarity_encoder.transform(curdf_test.values.reshape(-1, 1)))
            tempdf_heldtest = pd.DataFrame(similarity_encoder.transform(curdf_heldtest.values.reshape(-1, 1)))
            
            categ_cols_train = pd.concat([categ_cols_train,tempdf_train],axis=1)
            categ_cols_test = pd.concat([categ_cols_test,tempdf_test],axis=1)
            categ_cols_heldtest = pd.concat([categ_cols_heldtest,tempdf_heldtest],axis=1)
    #     elif y_cur[i] == 2:
    #         temp = pd.DataFrame()
    #         temp['month'] = dataDownstream.apply(lambda row: pd.Timestamp(row[curcol]).month, axis=1)
    #         print(temp)
    #         tempdf = pd.get_dummies(temp, columns=['month'])
    #         date_cols = pd.concat([date_cols,tempdf], axis=1, sort=False)
        if y_cur[i] == 3:
            arr_train = curdf_train.astype(str).values
            X_train = vec.fit_transform(arr_train)
            tempdf_train = pd.DataFrame(X_train.toarray())
            
            arr_test = curdf_test.astype(str).values
            X_test = vec.transform(arr_test)
            tempdf_test = pd.DataFrame(X_test.toarray())
            
            arr_heldtest = curdf_heldtest.astype(str).values
            X_heldtest = vec.transform(arr_heldtest)
            tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())            
            
            ngram_cols_train = pd.concat([ngram_cols_train,tempdf_train], axis=1, sort=False)
            ngram_cols_test = pd.concat([ngram_cols_test,tempdf_test], axis=1, sort=False)
            ngram_cols_heldtest = pd.concat([ngram_cols_heldtest,tempdf_heldtest], axis=1, sort=False)            
            
        if y_cur[i] == 4:
            temp_train = curdf_train.apply(lambda x: URLProcessor(x))
            arr_train = temp_train.astype(str).values
            X_train = vectorizerWord.fit_transform(arr_train)
            tempdf_train = pd.DataFrame(X_train.toarray())

            temp_test = curdf_test.apply(lambda x: URLProcessor(x))
            arr_test = temp_test.astype(str).values
            X_test = vectorizerWord.fit_transform(arr_test)
            tempdf_test = pd.DataFrame(X_test.toarray())
            
            temp_heldtest = curdf_heldtest.apply(lambda x: URLProcessor(x))
            arr_heldtest = temp_heldtest.astype(str).values
            X_heldtest = vectorizerWord.fit_transform(arr_heldtest)
            tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())            
            
            url_cols_train = pd.concat([url_cols_train,tempdf_train], axis=1, sort=False)
            url_cols_test = pd.concat([url_cols_test,tempdf_test], axis=1, sort=False)
            url_cols_heldtest = pd.concat([url_cols_heldtest,tempdf_heldtest], axis=1, sort=False)            
            
        if y_cur[i] in [2,3,5,6,8]:
            arr_train = curdf_train.astype(str).values
            X_train = vectorizer.fit_transform(arr_train)
            tempdf_train = pd.DataFrame(X_train.toarray())
            
            arr_test = curdf_test.astype(str).values
            X_test = vectorizer.transform(arr_test)
            tempdf_test = pd.DataFrame(X_test.toarray())
            
            arr_heldtest = curdf_heldtest.astype(str).values
            X_heldtest = vectorizer.transform(arr_heldtest)
            tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())
            
            ngram_cols_train = pd.concat([ngram_cols_train,tempdf_train], axis=1, sort=False)
            ngram_cols_test = pd.concat([ngram_cols_test,tempdf_test], axis=1, sort=False)
            ngram_cols_heldtest = pd.concat([ngram_cols_heldtest,tempdf_heldtest], axis=1, sort=False)            
            
    all_cols_train = pd.concat([all_cols_train,numeric_cols_train,categ_cols_train,ngram_cols_train], axis=1, sort=False)
    all_cols_test = pd.concat([all_cols_test,numeric_cols_test,categ_cols_test,ngram_cols_test], axis=1, sort=False)
    all_cols_heldtest = pd.concat([all_cols_heldtest,numeric_cols_heldtest,categ_cols_heldtest,ngram_cols_heldtest], axis=1, sort=False)
    
    return all_cols_train,all_cols_test,all_cols_heldtest


def FeaturizeH2O(dataDownstream_train,dataDownstream_test,dataDownstream_heldtest,attribute_names,y_cur):

    all_cols_train,numeric_cols_train,categ_cols_train,ngram_cols_train,url_cols_train = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    all_cols_test,numeric_cols_test,categ_cols_test,ngram_cols_test,url_cols_test = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    all_cols_heldtest,numeric_cols_heldtest,categ_cols_heldtest,ngram_cols_heldtest,url_cols_heldtest = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    
    vectorizer = CountVectorizer(ngram_range=(2,2),analyzer='char')
    vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
    vectorizerWord = CountVectorizer(ngram_range=(2,2))
    enc = OneHotEncoder(handle_unknown='ignore')
    
    for i in range(len(y_cur)):
        curcol = attribute_names[i]
        curdf_train = dataDownstream_train[curcol]
        curdf_test = dataDownstream_test[curcol]
        curdf_heldtest = dataDownstream_heldtest[curcol]
#         print(curcol)
        
        if y_cur[i] == 0:
            numeric_cols_train = pd.concat([numeric_cols_train,curdf_train],axis=1)
            numeric_cols_test = pd.concat([numeric_cols_test,curdf_test],axis=1)
            numeric_cols_heldtest = pd.concat([numeric_cols_heldtest,curdf_heldtest],axis=1)            
        if y_cur[i] in [1,10]:
            curdf_train.update('"' + curdf_train.astype(str) + '"')
            curdf_test.update('"' + curdf_test.astype(str) + '"')           
            curdf_heldtest.update('"' + curdf_heldtest.astype(str) + '"')
            
#             dataDownstream_h2o.update('"' + dataDownstream_h2o['RespondentID'].astype(str) + '"')
#             tempdf_train = pd.DataFrame(enc.fit_transform(curdf_train.to_frame()).toarray())
#             tempdf_test = pd.DataFrame(enc.transform(curdf_test.to_frame()).toarray())
#             tempdf_heldtest = pd.DataFrame(enc.transform(curdf_heldtest.to_frame()).toarray())
            
            categ_cols_train = pd.concat([categ_cols_train,curdf_train],axis=1)
            categ_cols_test = pd.concat([categ_cols_test,curdf_test],axis=1)
            categ_cols_heldtest = pd.concat([categ_cols_heldtest,curdf_heldtest],axis=1)
            
        if y_cur[i] == 3:
            arr_train = curdf_train.astype(str).values
            X_train = vec.fit_transform(arr_train)
            tempdf_train = pd.DataFrame(X_train.toarray())
            
            arr_test = curdf_test.astype(str).values
            X_test = vec.transform(arr_test)
            tempdf_test = pd.DataFrame(X_test.toarray())
            
            arr_heldtest = curdf_heldtest.astype(str).values
            X_heldtest = vec.transform(arr_heldtest)
            tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())            
            
            ngram_cols_train = pd.concat([ngram_cols_train,tempdf_train], axis=1, sort=False)
            ngram_cols_test = pd.concat([ngram_cols_test,tempdf_test], axis=1, sort=False)
            ngram_cols_heldtest = pd.concat([ngram_cols_heldtest,tempdf_heldtest], axis=1, sort=False)            
            
            for col in ngram_cols_train.columns: ngram_cols_train.rename(columns={col: "NG3_" + str(col)},inplace=True)
            for col in ngram_cols_test.columns: ngram_cols_test.rename(columns={col: "NG3_" + str(col)},inplace=True)
            for col in ngram_cols_heldtest.columns: ngram_cols_heldtest.rename(columns={col: "NG3_" + str(col)},inplace=True)     
            
        if y_cur[i] == 4:
            temp_train = curdf_train.apply(lambda x: URLProcessor(x))
            arr_train = temp_train.astype(str).values
            X_train = vectorizerWord.fit_transform(arr_train)
            tempdf_train = pd.DataFrame(X_train.toarray())

            temp_test = curdf_test.apply(lambda x: URLProcessor(x))
            arr_test = temp_test.astype(str).values
            X_test = vectorizerWord.fit_transform(arr_test)
            tempdf_test = pd.DataFrame(X_test.toarray())
            
            temp_heldtest = curdf_heldtest.apply(lambda x: URLProcessor(x))
            arr_heldtest = temp_heldtest.astype(str).values
            X_heldtest = vectorizerWord.fit_transform(arr_heldtest)
            tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())            
            
            url_cols_train = pd.concat([url_cols_train,tempdf_train], axis=1, sort=False)
            url_cols_test = pd.concat([url_cols_test,tempdf_test], axis=1, sort=False)
            url_cols_heldtest = pd.concat([url_cols_heldtest,tempdf_heldtest], axis=1, sort=False)            
            
        if y_cur[i] in [2,3,5,6,8]:
            arr_train = curdf_train.astype(str).values
            X_train = vectorizer.fit_transform(arr_train)
            tempdf_train = pd.DataFrame(X_train.toarray())
            
            arr_test = curdf_test.astype(str).values
            X_test = vectorizer.transform(arr_test)
            tempdf_test = pd.DataFrame(X_test.toarray())
            
            arr_heldtest = curdf_heldtest.astype(str).values
            X_heldtest = vectorizer.transform(arr_heldtest)
            tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())
            
            ngram_cols_train = pd.concat([ngram_cols_train,tempdf_train], axis=1, sort=False)
            ngram_cols_test = pd.concat([ngram_cols_test,tempdf_test], axis=1, sort=False)
            ngram_cols_heldtest = pd.concat([ngram_cols_heldtest,tempdf_heldtest], axis=1, sort=False)            

            for col in ngram_cols_train.columns: ngram_cols_train.rename(columns={col: "NG_" + str(col)},inplace=True)
            for col in ngram_cols_test.columns: ngram_cols_test.rename(columns={col: "NG_" + str(col)},inplace=True)
            for col in ngram_cols_heldtest.columns: ngram_cols_heldtest.rename(columns={col: "NG_" + str(col)},inplace=True)       
            
    all_cols_train = pd.concat([all_cols_train,numeric_cols_train,categ_cols_train,ngram_cols_train], axis=1, sort=False)
    all_cols_test = pd.concat([all_cols_test,numeric_cols_test,categ_cols_test,ngram_cols_test], axis=1, sort=False)
    all_cols_heldtest = pd.concat([all_cols_heldtest,numeric_cols_heldtest,categ_cols_heldtest,ngram_cols_heldtest], axis=1, sort=False)
    
    return all_cols_train,all_cols_test,all_cols_heldtest


def FeaturizeH2OSimiarity(dataDownstream_train,dataDownstream_test,dataDownstream_heldtest,attribute_names,y_cur):

    all_cols_train,numeric_cols_train,categ_cols_train,ngram_cols_train,url_cols_train = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    all_cols_test,numeric_cols_test,categ_cols_test,ngram_cols_test,url_cols_test = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    all_cols_heldtest,numeric_cols_heldtest,categ_cols_heldtest,ngram_cols_heldtest,url_cols_heldtest = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    
    vectorizer = CountVectorizer(ngram_range=(2,2),analyzer='char')
    vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
    vectorizerWord = CountVectorizer(ngram_range=(2,2))
    enc = OneHotEncoder(handle_unknown='ignore')
    similarity_encoder = SimilarityEncoder(similarity='ngram')
    
    for i in range(len(y_cur)):

        curcol = attribute_names[i]
        curdf_train = dataDownstream_train[curcol]
        curdf_test = dataDownstream_test[curcol]
        curdf_heldtest = dataDownstream_heldtest[curcol]
        
        if y_cur[i] == 0:
            numeric_cols_train = pd.concat([numeric_cols_train,curdf_train],axis=1)
            numeric_cols_test = pd.concat([numeric_cols_test,curdf_test],axis=1)
            numeric_cols_heldtest = pd.concat([numeric_cols_heldtest,curdf_heldtest],axis=1)
        if y_cur[i] in [1]:
            curdf_train.update('"' + curdf_train.astype(str) + '"')
            curdf_test.update('"' + curdf_test.astype(str) + '"')           
            curdf_heldtest.update('"' + curdf_heldtest.astype(str) + '"')
            
            categ_cols_train = pd.concat([categ_cols_train,curdf_train],axis=1)
            categ_cols_test = pd.concat([categ_cols_test,curdf_test],axis=1)
            categ_cols_heldtest = pd.concat([categ_cols_heldtest,curdf_heldtest],axis=1)           
        if y_cur[i] in [10]:
            curdf_train.update('"' + curdf_train.astype(str) + '"')
            curdf_test.update('"' + curdf_test.astype(str) + '"')           
            curdf_heldtest.update('"' + curdf_heldtest.astype(str) + '"')            
            
            tempdf_train = pd.DataFrame(similarity_encoder.fit_transform(curdf_train.values.reshape(-1, 1)))
            tempdf_test = pd.DataFrame(similarity_encoder.transform(curdf_test.values.reshape(-1, 1)))
            tempdf_heldtest = pd.DataFrame(similarity_encoder.transform(curdf_heldtest.values.reshape(-1, 1)))
            
            categ_cols_train = pd.concat([categ_cols_train,tempdf_train],axis=1)
            categ_cols_test = pd.concat([categ_cols_test,tempdf_test],axis=1)
            categ_cols_heldtest = pd.concat([categ_cols_heldtest,tempdf_heldtest],axis=1)
          
            for col in categ_cols_train.columns: categ_cols_train.rename(columns={col: "DUP_" + str(col) + "_" + str(i)},inplace=True)
            for col in categ_cols_test.columns: categ_cols_test.rename(columns={col: "DUP_" + str(col) + "_" + str(i)},inplace=True)
            for col in categ_cols_heldtest.columns: categ_cols_heldtest.rename(columns={col: "DUP_" + str(col) + "_" + str(i)},inplace=True)       
            
    #     elif y_cur[i] == 2:
    #         temp = pd.DataFrame()
    #         temp['month'] = dataDownstream.apply(lambda row: pd.Timestamp(row[curcol]).month, axis=1)
    #         print(temp)
    #         tempdf = pd.get_dummies(temp, columns=['month'])
    #         date_cols = pd.concat([date_cols,tempdf], axis=1, sort=False)
        if y_cur[i] == 3:
            arr_train = curdf_train.astype(str).values
            X_train = vec.fit_transform(arr_train)
            tempdf_train = pd.DataFrame(X_train.toarray())
            
            arr_test = curdf_test.astype(str).values
            X_test = vec.transform(arr_test)
            tempdf_test = pd.DataFrame(X_test.toarray())
            
            arr_heldtest = curdf_heldtest.astype(str).values
            X_heldtest = vec.transform(arr_heldtest)
            tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())            
            
            ngram_cols_train = pd.concat([ngram_cols_train,tempdf_train], axis=1, sort=False)
            ngram_cols_test = pd.concat([ngram_cols_test,tempdf_test], axis=1, sort=False)
            ngram_cols_heldtest = pd.concat([ngram_cols_heldtest,tempdf_heldtest], axis=1, sort=False)            
            
        if y_cur[i] == 4:
            temp_train = curdf_train.apply(lambda x: URLProcessor(x))
            arr_train = temp_train.astype(str).values
            X_train = vectorizerWord.fit_transform(arr_train)
            tempdf_train = pd.DataFrame(X_train.toarray())

            temp_test = curdf_test.apply(lambda x: URLProcessor(x))
            arr_test = temp_test.astype(str).values
            X_test = vectorizerWord.fit_transform(arr_test)
            tempdf_test = pd.DataFrame(X_test.toarray())
            
            temp_heldtest = curdf_heldtest.apply(lambda x: URLProcessor(x))
            arr_heldtest = temp_heldtest.astype(str).values
            X_heldtest = vectorizerWord.fit_transform(arr_heldtest)
            tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())            
            
            url_cols_train = pd.concat([url_cols_train,tempdf_train], axis=1, sort=False)
            url_cols_test = pd.concat([url_cols_test,tempdf_test], axis=1, sort=False)
            url_cols_heldtest = pd.concat([url_cols_heldtest,tempdf_heldtest], axis=1, sort=False)

        # if y_cur[i] == 5:
        #     arr_train = curdf_train.astype(str).values
        #     X_train = vectorizer.fit_transform(arr_train)
        #     tempdf_train = pd.DataFrame(X_train.toarray())
            
        #     arr_test = curdf_test.astype(str).values
        #     X_test = vectorizer.transform(arr_test)
        #     tempdf_test = pd.DataFrame(X_test.toarray())
            
        #     arr_heldtest = curdf_heldtest.astype(str).values
        #     X_heldtest = vectorizer.transform(arr_heldtest)
        #     tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())
            
        #     ngram_cols_train = pd.concat([ngram_cols_train,tempdf_train], axis=1, sort=False)
        #     ngram_cols_test = pd.concat([ngram_cols_test,tempdf_test], axis=1, sort=False)
        #     ngram_cols_heldtest = pd.concat([ngram_cols_heldtest,tempdf_heldtest], axis=1, sort=False)            
            
        #     for col in ngram_cols_train.columns: ngram_cols_train.rename(columns={col: "EN_" + str(col) + "_" + str(i)},inplace=True)
        #     for col in ngram_cols_test.columns: ngram_cols_test.rename(columns={col: "EN_" + str(col) + "_" + str(i)},inplace=True)
        #     for col in ngram_cols_heldtest.columns: ngram_cols_heldtest.rename(columns={col: "EN_" + str(col) + "_" + str(i)},inplace=True)  

            
        if y_cur[i] in [2,3,5, 6,8]:

            # print(i)
            # print('---------abc')

            arr_train = curdf_train.astype(str).values
            X_train = vectorizer.fit_transform(arr_train)
            tempdf_train = pd.DataFrame(X_train.toarray())
            
            arr_test = curdf_test.astype(str).values
            X_test = vectorizer.transform(arr_test)
            tempdf_test = pd.DataFrame(X_test.toarray())
            
            arr_heldtest = curdf_heldtest.astype(str).values
            X_heldtest = vectorizer.transform(arr_heldtest)
            tempdf_heldtest = pd.DataFrame(X_heldtest.toarray())

            for col in tempdf_train.columns: tempdf_train.rename(columns={col: str(col) + "_NG_" + str(i)},inplace=True)
            for col in tempdf_test.columns: tempdf_test.rename(columns={col: str(col) + "_NG_" + str(i)},inplace=True)
            for col in tempdf_heldtest.columns: tempdf_heldtest.rename(columns={col: str(col) + "_NG_" + str(i)},inplace=True)       
            
            ngram_cols_train = pd.concat([ngram_cols_train,tempdf_train], axis=1, sort=False)
            ngram_cols_test = pd.concat([ngram_cols_test,tempdf_test], axis=1, sort=False)
            ngram_cols_heldtest = pd.concat([ngram_cols_heldtest,tempdf_heldtest], axis=1, sort=False)

            # print(ngram_cols_train.columns)
            # print(len(ngram_cols_train.columns))
            # print(ngram_cols_heldtest.columns)  
            # print(len(ngram_cols_heldtest.columns))
            # print()    
            
    all_cols_train = pd.concat([all_cols_train,numeric_cols_train,categ_cols_train,ngram_cols_train], axis=1, sort=False)
    all_cols_test = pd.concat([all_cols_test,numeric_cols_test,categ_cols_test,ngram_cols_test], axis=1, sort=False)
    all_cols_heldtest = pd.concat([all_cols_heldtest,numeric_cols_heldtest,categ_cols_heldtest,ngram_cols_heldtest], axis=1, sort=False)
    
    return all_cols_train,all_cols_test,all_cols_heldtest