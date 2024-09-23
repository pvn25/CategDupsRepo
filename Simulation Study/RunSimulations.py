#Copyright 2024 Vraj Shah, Thomas Parashos, and Arun Kumar
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys, imp, math, random, time
import numpy as np
import pandas as pd
import csv
import random, copy
from pathlib import Path

import sys, imp, math, random
from cStringIO import StringIO

from io import StringIO
from sklearn import tree, metrics
from sklearn import feature_selection
import sklearn
from downstream_model import *
from Featurize import *

from matplotlib import pyplot as plt
from random import choices
import itertools

import warnings
warnings.filterwarnings('ignore')

import ast
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.svm import SVC

plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.autolayout': True})

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


np.random.seed(100)
random.seed(100)

def writecsvfile(fname,data):
    mycsv = csv.writer(open(fname,'wb'), quoting=csv.QUOTE_ALL)
    for row in data:
        mycsv.writerow(row)

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def onehotvector(x,k):
    lst = [0] * k
    lst[x] = 1
    return lst


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def createCPT(DomainSize):
    RandNumLst = []
    for i in range(DomainSize):
        randNum = random.uniform(0, 1)
        RandNumLst.append(randNum)
    return RandNumLst

def createCPTGamma(DomainSize):
    p = 0.9
    gamma = 0.99
    
    RandNumLst = []
    for i in range(DomainSize):
        randNum = p * (gamma ** i)
        RandNumLst.append(randNum)
    return RandNumLst

def createCPTBeta(DomainSize):
    RandNumLst = []
    for i in range(DomainSize):
        randNum = np.random.beta(5,2)
        RandNumLst.append(randNum)
    return RandNumLst

def SampleZipf(nr, dxr, NumEnts2sample):
    a = 2
    s = np.random.zipf(a, nr)
    
    vals = dxr
    count, bins, ignored = plt.hist(s[s<vals], vals, density=True)
    plt.close()
    
    count_lst = count.tolist()
    
    sum_freq = sum(count_lst)

    count_lst = [x*1.0/sum_freq for x in count_lst]
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar([i for i in range(1,dxr+1)], count_lst)
    plt.xticks([i for i in range(1,dxr+1)])
#     plt.show()
    plt.close()
    
    tmp = np.random.choice(a = dxr, size = NumEnts2sample, p = count_lst)
    sampled_vals = [x+1 for x in tmp]

    
    return sampled_vals

def CreateNDCPT(n,dxr):
    
    global dr
    cptdic = {}

    if dr == 3:
    
        j=0
        while j < dxr:
            for k in range(dxr):
                tmpstr = ''
                for l in range(dxr):
                    tmpstr = str(j) + '#'
                    tmpstr = tmpstr + str(k) + '#'                    
                    tmpstr = tmpstr + str(l) + '#'
                    randNum = random.uniform(0, 1)
                    if randNum > 0.5: randNum = 1
                    else: randNum = 0
                    cptdic[tmpstr] = randNum
            j += 1    
    
    return cptdic

def AblationStudy(dataDownstream, y, y_cur, attribute_names, scrtr_LR, scrval_LR, scrte_LR, scrtr_RF, scrval_RF, scrte_RF):

    dataDownstreamX = copy.deepcopy(dataDownstream)   

    truthModel_LR, truth_train_LR, truth_val_LR, truth_test_LR = 'lrm', scrtr_LR, scrval_LR, scrte_LR
    truthModel_RF, truth_train_RF, truth_val_RF, truth_test_RF = 'rfm', scrtr_RF, scrval_RF, scrte_RF

    print(truth_train_LR, truth_val_LR, truth_test_LR)
    print(truth_train_RF, truth_val_RF, truth_test_RF)

    dataDownstream_dedup_ablation = copy.deepcopy(dataDownstreamX)

    attribute_names_ablation = []
    y_cur_ablation = []

    for i in range(len(attribute_names)):
        if y_cur[i] != 7:
            attribute_names_ablation.append(attribute_names[i])
            y_cur_ablation.append(y_cur[i])

    print(attribute_names_ablation)
    print(y_cur_ablation)

    dataDownstream_dedup_ablation = dataDownstream_dedup_ablation[attribute_names_ablation]

    models_LR, trainscores_LR, valscores_LR, testscores_LR = [],[],[],[]
    models_RF, trainscores_RF, valscores_RF, testscores_RF = [],[],[],[]

    for i in range(len(y_cur_ablation)):
        print(y_cur_ablation[i])

        attribute_names_moving, y_cur_moving = copy.deepcopy(attribute_names_ablation),copy.deepcopy(y_cur_ablation)
        curcol = attribute_names_moving[i]
        curdf = dataDownstream_dedup_ablation.drop(curcol, axis = 1)

        attribute_names_moving.pop(i)
        y_cur_moving.pop(i)

        print('***')
        print(curcol)
        print(attribute_names_moving)
        ########################################
        bestPerformingModel_LR, avgsc_train_lst_LR, avgsc_lst_LR, avgsc_hld_lst_LR = LogRegClassifier(curdf,y, y_cur_moving,attribute_names_moving,0)
        bestPerformingModel_RF, avgsc_train_lst_RF, avgsc_lst_RF, avgsc_hld_lst_RF = RandForestClassifier(curdf,y, y_cur_moving,attribute_names_moving,0)
        ########################################
        trainscores_LR.append(np.mean(avgsc_train_lst_LR))
        valscores_LR.append(np.mean(avgsc_lst_LR))
        testscores_LR.append(np.mean(avgsc_hld_lst_LR))
        ########################################
        trainscores_RF.append(np.mean(avgsc_train_lst_RF))
        valscores_RF.append(np.mean(avgsc_lst_RF))
        testscores_RF.append(np.mean(avgsc_hld_lst_RF))
        ########################################    
    
    print(trainscores_LR, valscores_LR, testscores_LR)
    print(trainscores_RF, valscores_RF, testscores_RF)
    
    drops_by_attribute_LR,drops_by_attribute_RF,drops_by_attribute_h2o = [],[],[]
    for i in range(len(attribute_names_ablation)):
        print(attribute_names_ablation[i])

        drop_LR = (truth_test_LR - testscores_LR[i]*100)
        print(drop_LR)
        drops_by_attribute_LR.append(drop_LR)

        drop_RF = (truth_test_RF - testscores_RF[i]*100)
        print(drop_RF)
        drops_by_attribute_RF.append(drop_RF) 

        print('\n')    
    
    df = pd.DataFrame()
    df['attribute_names'] = attribute_names_ablation
    df['drop_LR'] = drops_by_attribute_LR
    df['drop_RF'] = drops_by_attribute_RF
    
    LR_index = (np.argsort(-np.array(drops_by_attribute_LR))+1).tolist()
    RF_index = (np.argsort(-np.array(drops_by_attribute_RF))+1).tolist()    
    
    return df,LR_index,RF_index

def getY(rtup_lst,dxr):
    global CPTdic, dr
    
    tmpstr = ''
    for j in range(len(rtup_lst)):
#         if j == dr-1: break
        tmpstr = tmpstr + str(int(rtup_lst[j])) + '#'    
    
    prob0 = CPTdic[tmpstr]
    
    thisy = np.random.choice([0,1],1,p=[prob0,1-prob0]).tolist()[0]
    return thisy

def buildDataset(trainsetx, trainsetx_Number, trainsetx_both, trainsety, rtup_lst, DomainLst,OHElst,dxr):
    global ProbLst, dr
    
    dval = copy.deepcopy(rtup_lst)

    thisy = getY(rtup_lst,dxr)
    dval_Number = dval
    
    trainsetx_Number = trainsetx_Number + [dval_Number]
    trainsety = trainsety + [thisy]
    
    return 'trainsetx', trainsetx_Number, '', trainsety

def DatasetToDataFrame(trainsetx, trainsety):
    global dupColIndex
    dataDownstream = pd.DataFrame(trainsetx)
    dataDownstream.rename(columns={ dataDownstream.columns[dupColIndex]: "duplicateColumn" }, inplace = True)

    attribute_names = dataDownstream.columns.tolist()

    if dupColIndex == 0:
        y_cur = [10]
        for x in range(len(attribute_names)-1): y_cur.append(1)

    if dupColIndex == 2:            
        y_cur = []
        for x in range(len(attribute_names)-1): y_cur.append(1)
        y_cur.append(10)
        
    if dupColIndex == 3:            
        y_cur = [1,1,1,10]
        
    y =  pd.DataFrame(trainsety, columns = ['y'])

    return dataDownstream, y, y_cur, attribute_names


def IntroduceDirtiness(dfAblation, dataDownstream, attribute_names, dxr, PERC_OCCUR, scrtr_LR, scrval_LR, scrte_LR, scrtr_RF, scrval_RF, scrte_RF):
    global SAVEPLACE, PERC_ENTY
    categcols = ['duplicateColumn']
    
    ### p1_perc --> Percentage of entities that have duplicates
    ### noise --> Percentage of occurences that are diluted with duplicate values    
    
    GRP_SIZE_MINUS_1 = 1
    PERC_OCCUR = PERC_OCCUR*0.01
    NUM_DIRTY_DT = 10

    for curcol in categcols:
        curdic = dict(dataDownstream[curcol].value_counts())

        p1_perc = [PERC_ENTY]
        for CURPERC in p1_perc:
            print('CURPERC value is:' + str(CURPERC))
            
            tmp = ((dataDownstream[curcol].nunique())*CURPERC)*1.0/100
            NENT = int(round(tmp,0))            

            lst_vals = list(curdic.keys())
            
            print('Dictionary keys:')
            print(lst_vals)            
            
            if CURPERC == 100:
                NENT = dataDownstream[curcol].nunique()
                possible_combinations = [random.sample(lst_vals,NENT)]
            else:
                possible_combinations = []
                dic_of_strlsts = {}
                
                while True:
                    if len(dic_of_strlsts) == NUM_DIRTY_DT: break
                    tmplst = random.sample(lst_vals, NENT)

                    if str(sorted(tmplst)) not in dic_of_strlsts:
                        dic_of_strlsts[str(sorted(tmplst))] = 1
                        possible_combinations.append(tmplst)     
         
                print('All possible_combinations:')
                print(possible_combinations)  
                   

            print('NENT value is:' + str(NENT))

            noise = PERC_OCCUR
            indexlstlst = []

            bestPerformingModel_LR_full, avgsc_train_lst_LR_full, avgsc_lst_LR_full, avgsc_hld_lst_LR_full = [],[],[],[]
            bestPerformingModel_RF_full, avgsc_train_lst_RF_full, avgsc_lst_RF_full, avgsc_hld_lst_RF_full = [],[],[],[]

            k = 0
            for comb in possible_combinations:
#                 if k > 1: break
                k = k + 1
                print(comb)
                curdataDownstream = copy.deepcopy(dataDownstream)

                for pq in comb:
#                     print('\nCurrent value is:' + str(pq))
    
                    dirtiness = int(curdic[pq]*noise)
#                     print(dirtiness)
                    abc = curdataDownstream[curdataDownstream[curcol] == pq].sample(dirtiness, random_state=1)
                    indxlst = abc.index.tolist()
                    indexlstlst.append(indxlst)

                    chk_indxlst = chunkIt(indxlst, GRP_SIZE_MINUS_1)

                    for j in range(len(chk_indxlst)): curdataDownstream.loc[chk_indxlst[j], curcol] = str(pq) + '_' + str(j) + '_dummy'    

                print(curdataDownstream[curcol].value_counts())
                print('\n')         

                bestPerformingModel_LR,avgsc_train_lst_LR,avgsc_lst_LR,avgsc_hld_lst_LR = LogRegClassifier(curdataDownstream, y, y_cur, attribute_names, 0)
                bestPerformingModel_RF,avgsc_train_lst_RF,avgsc_lst_RF,avgsc_hld_lst_RF = RandForestClassifier(curdataDownstream, y, y_cur, attribute_names, 0)
        
                avgsc_train_lst_LR,avgsc_lst_LR,avgsc_hld_lst_LR = round(avgsc_train_lst_LR*100.0,3), round(avgsc_lst_LR*100.0,3), round(avgsc_hld_lst_LR*100.0,3)
                avgsc_train_lst_RF,avgsc_lst_RF,avgsc_hld_lst_RF = round(avgsc_train_lst_RF*100.0,3), round(avgsc_lst_RF*100.0,3), round(avgsc_hld_lst_RF*100.0,3)
        
                bestPerformingModel_LR_full.append(bestPerformingModel_LR)
                avgsc_train_lst_LR_full.append(avgsc_train_lst_LR)
                avgsc_lst_LR_full.append(avgsc_lst_LR)
                avgsc_hld_lst_LR_full.append(avgsc_hld_lst_LR)

                bestPerformingModel_RF_full.append(bestPerformingModel_RF)
                avgsc_train_lst_RF_full.append(avgsc_train_lst_RF)
                avgsc_lst_RF_full.append(avgsc_lst_RF)
                avgsc_hld_lst_RF_full.append(avgsc_hld_lst_RF)

#             diff_train_lst_LR_full, diff_lst_LR_full, diff_hld_lst_LR_full, diff_train_lst_RF_full, diff_lst_RF_full, diff_hld_lst_RF_full = [], [], [], [], [], []
            
            diff_train_lst_LR_full = [round(scrtr_LR - number,3) for number in avgsc_train_lst_LR_full]
            diff_lst_LR_full = [round(scrval_LR - number,3) for number in avgsc_lst_LR_full] 
            diff_hld_lst_LR_full = [round(scrte_LR - number,3) for number in avgsc_hld_lst_LR_full]
            
            diff_train_lst_RF_full = [round(scrtr_RF - number,3) for number in avgsc_train_lst_RF_full]
            diff_lst_RF_full = [round(scrval_RF - number,3) for number in avgsc_lst_RF_full]
            diff_hld_lst_RF_full = [round(scrte_RF - number,3) for number in avgsc_hld_lst_RF_full]

#             curWD = 'logs/synthetic/duplicates-' + str(GRP_SIZE_MINUS_1)
#             Path(curWD).mkdir(parents=True, exist_ok=True)
            
#             sample = open(curWD + '/' + str(curcol) + '-' + str(CURPERC) + '.txt', 'a')
            sample = open(SAVEPLACE, 'a')
            print('Original Dataset:', file = sample)
            print('---', file = sample)
            print(scrtr_LR, file = sample)
            print(scrval_LR, file = sample)
            print(scrte_LR, file = sample)

            print('---', file = sample)
            print(scrtr_RF, file = sample)
            print(scrval_RF, file = sample)
            print(scrte_RF, file = sample)
#             print('\n', file = sample)            
            
            print('---', file = sample)            
            print(dfAblation, file = sample)
            print('\n', file = sample)             
            
            print('Categories that are diluted with duplicates:', file = sample)
            print(possible_combinations[:10], file = sample)

            print('---', file = sample)
            print(avgsc_train_lst_LR_full, file = sample)
            print(avgsc_lst_LR_full, file = sample)
            print(avgsc_hld_lst_LR_full, file = sample)

            print('---', file = sample)
            print(avgsc_train_lst_RF_full, file = sample)
            print(avgsc_lst_RF_full, file = sample)
            print(avgsc_hld_lst_RF_full, file = sample)
            print('\n', file = sample)
            
            print('Difference between Orignal and Duplicate Dataset:', file = sample)
            print('---', file = sample)
            print(diff_train_lst_LR_full, file = sample)
            print(diff_lst_LR_full, file = sample)
            print(diff_hld_lst_LR_full, file = sample)

            print('---', file = sample)
            print(diff_train_lst_RF_full, file = sample)
            print(diff_lst_RF_full, file = sample)
            print(diff_hld_lst_RF_full, file = sample)
            print('\n', file = sample)              
            
            sample.close()
    
    mean_train_acc_LR, mean_val_acc_LR, mean_test_acc_LR = np.mean(avgsc_train_lst_LR_full), np.mean(avgsc_lst_LR_full), np.mean(avgsc_hld_lst_LR_full)
    mean_train_acc_RF, mean_val_acc_RF, mean_test_acc_RF = np.mean(avgsc_train_lst_RF_full), np.mean(avgsc_lst_RF_full), np.mean(avgsc_hld_lst_RF_full)

    return mean_train_acc_LR, mean_val_acc_LR, mean_test_acc_LR, mean_train_acc_RF, mean_val_acc_RF, mean_test_acc_RF    
#     return avgsc_train_lst_LR_full, avgsc_lst_LR_full, avgsc_hld_lst_LR_full, avgsc_train_lst_RF_full, avgsc_lst_RF_full, avgsc_hld_lst_RF_full

def SVMClassifierr(dataDownstream, y, y_cur,attribute_names,similarity_flag):
    X_train, X_test_df,y_train,y_test = train_test_split(dataDownstream,y, test_size=0.2,random_state=Hcurstate, shuffle=False)
    X_train_cur_df, X_test_cur_df, y_train_cur, y_test_cur = train_test_split(X_train,y_train, test_size=0.25,random_state=Hcurstate, shuffle=False)

    # X_train = X_train.reset_index(drop=True)
    # y_train = y_train.reset_index(drop=True)

    X_test_df = X_test_df.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_test = y_test.values

    X_train_cur_df = X_train_cur_df.reset_index(drop=True)
    y_train_cur = y_train_cur.reset_index(drop=True)
    y_train_cur = y_train_cur.values

    X_test_cur_df = X_test_cur_df.reset_index(drop=True)
    y_test_cur = y_test_cur.reset_index(drop=True)
    y_test_cur = y_test_cur.values


    # print('train:')
    # print(X_train_cur_df)
    # print('val:')        
    # print(X_test_cur_df)
    # print('test:')        
    # print(X_test_df)
    
    # print('train y:')
    # print(y_train_cur)

    # val_arr = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
    val_arr = [0.0001,0.001,0.01,0.1,1,10]

    avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
    avgsc,avgsc_train,avgsc_hld = 0,0,0

    if similarity_flag: all_cols_train,all_cols_test,all_cols_heldtest = FeaturizeSimiarity(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)
    else: all_cols_train,all_cols_test,all_cols_heldtest = Featurize(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)

#         print('train:')        
    print(all_cols_train.shape)
    # print(all_cols_train)

#     bestPerformingModel = LogisticRegression(penalty='l2',C = 1,random_state=Hcurstate)
    # bestPerformingModel = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=100 , random_state=Hcurstate)
    # bestPerformingModel = bestPerformingModel.fit(all_cols_train, y_train_cur)                    
    
    bestPerformingModel = SVC(kernel='rbf',C = 1,random_state=Hcurstate)

    bestscore = 0
    for val in val_arr:
        # clf = LogisticRegression(penalty='l2',C = val,random_state=Hcurstate)
        clf = SVC(kernel='rbf',C = val,random_state=Hcurstate)
        clf.fit(all_cols_train, y_train_cur)
        sc = clf.score(all_cols_test, y_test_cur)

        if bestscore < sc:
            bestscore = sc
            bestPerformingModel = clf

    bscr_train = bestPerformingModel.score(all_cols_train, y_train_cur)
    bscr = bestPerformingModel.score(all_cols_test, y_test_cur)
    bscr_hld = bestPerformingModel.score(all_cols_heldtest, y_test)

    avgsc_train_lst.append(bscr_train)
    avgsc_lst.append(bscr)
    avgsc_hld_lst.append(bscr_hld)

    avgsc_train = avgsc_train + bscr_train    
    avgsc = avgsc + bscr
    avgsc_hld = avgsc_hld + bscr_hld

    print(bscr_train)
    print(bscr)
    print(bscr_hld)

    return bestPerformingModel, avgsc_train_lst[0],avgsc_lst[0],avgsc_hld_lst[0]

def XGBClassifierr(dataDownstream, y, y_cur,attribute_names,similarity_flag):
    X_train, X_test_df,y_train,y_test = train_test_split(dataDownstream,y, test_size=0.2,random_state=Hcurstate, shuffle=False)
    X_train_cur_df, X_test_cur_df, y_train_cur, y_test_cur = train_test_split(X_train,y_train, test_size=0.25,random_state=Hcurstate, shuffle=False)

    # X_train = X_train.reset_index(drop=True)
    # y_train = y_train.reset_index(drop=True)

    X_test_df = X_test_df.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_test = y_test.values

    X_train_cur_df = X_train_cur_df.reset_index(drop=True)
    y_train_cur = y_train_cur.reset_index(drop=True)
    y_train_cur = y_train_cur.values

    X_test_cur_df = X_test_cur_df.reset_index(drop=True)
    y_test_cur = y_test_cur.reset_index(drop=True)
    y_test_cur = y_test_cur.values


    # print('train:')
    # print(X_train_cur_df)
    # print('val:')        
    # print(X_test_cur_df)
    # print('test:')        
    # print(X_test_df)
    
    # print('train y:')
    # print(y_train_cur)

    # val_arr = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
    # val_arr = [0.0001,0.001,0.01,0.1,1,10]
    n_estimators_grid = [5,25,50,100]
    max_depth_grid = [5,25,50,100]  

    avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
    avgsc,avgsc_train,avgsc_hld = 0,0,0

    if similarity_flag: all_cols_train,all_cols_test,all_cols_heldtest = FeaturizeSimiarity(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)
    else: all_cols_train,all_cols_test,all_cols_heldtest = Featurize(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)

#         print('train:')        
    print(all_cols_train.shape)
    # print(all_cols_train)

#     bestPerformingModel = LogisticRegression(penalty='l2',C = 1,random_state=Hcurstate)
    # bestPerformingModel = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=100 , random_state=Hcurstate)
    # bestPerformingModel = bestPerformingModel.fit(all_cols_train, y_train_cur)                    
    
    # bestPerformingModel = SVC(kernel='rbf',C = 1,random_state=Hcurstate)
    bestPerformingModel = XGBClassifier(max_depth = 50,n_estimators=50, random_state=Hcurstate)
    all_cols_train = np.array(all_cols_train)
    all_cols_test = np.array(all_cols_test)

    bestPerformingModel.fit(all_cols_train, y_train_cur)
    sc = bestPerformingModel.score(all_cols_test, y_test_cur)
    print("Current Score:", sc)
    
#     bestscore = 0
#     for ne in n_estimators_grid:
#         for md in max_depth_grid:
# #             clf = RandomForestClassifier(n_estimators=ne,max_depth=md, random_state=Hcurstate)
#             clf = XGBClassifier(max_depth = md,n_estimators=ne, random_state=Hcurstate)
#             all_cols_train = np.array(all_cols_train)
#             all_cols_test = np.array(all_cols_test)

#             clf.fit(all_cols_train, y_train_cur)
#             sc = clf.score(all_cols_test, y_test_cur)
#             print("NE:MD", ne,md)
#             print("Current Score:", sc)
#             if bestscore < sc:
#                 bestscore = sc
#                 bestPerformingModel = clf        
    
    # bestscore = 0
    # for val in val_arr:
    #     # clf = LogisticRegression(penalty='l2',C = val,random_state=Hcurstate)
    #     clf = SVC(kernel='rbf',C = val,random_state=Hcurstate)
    #     clf.fit(all_cols_train, y_train_cur)
    #     sc = clf.score(all_cols_test, y_test_cur)

    #     if bestscore < sc:
    #         bestscore = sc
    #         bestPerformingModel = clf

    bscr_train = bestPerformingModel.score(all_cols_train, y_train_cur)
    bscr = bestPerformingModel.score(all_cols_test, y_test_cur)
    bscr_hld = bestPerformingModel.score(all_cols_heldtest, y_test)

    avgsc_train_lst.append(bscr_train)
    avgsc_lst.append(bscr)
    avgsc_hld_lst.append(bscr_hld)

    avgsc_train = avgsc_train + bscr_train    
    avgsc = avgsc + bscr
    avgsc_hld = avgsc_hld + bscr_hld

    print(bscr_train)
    print(bscr)
    print(bscr_hld)

    return bestPerformingModel, avgsc_train_lst[0],avgsc_lst[0],avgsc_hld_lst[0]


def IntroduceDirtinessSkew(dfAblation, dataDownstream, attribute_names, dxr, PERC_OCCUR, scrtr_LR, scrval_LR, scrte_LR, scrtr_RF, scrval_RF, scrte_RF, mse_LR_TR, bias_LR_TR, var_LR_TR, mse_RF_TR, bias_RF_TR, var_RF_TR):
    global SAVEPLACE, PERC_ENTY
    categcols = ['duplicateColumn']
    
    ### p1_perc --> Percentage of entities that have duplicates
    ### noise --> Percentage of occurences that are diluted with duplicate values    
    
    GRP_SIZE_MINUS_1 = 1
    PERC_OCCUR = PERC_OCCUR*0.01
    NUM_DIRTY_DT = 3

    for curcol in categcols:
        curdic = dict(dataDownstream[curcol].value_counts())

        p1_perc = [PERC_ENTY]
        for CURPERC in p1_perc:
            print('CURPERC value is:' + str(CURPERC))
            
            tmp = ((dataDownstream[curcol].nunique())*CURPERC)*1.0/100
            NENT = int(round(tmp,0))            

            lst_vals = list(curdic.keys())
            
            print('Dictionary keys:')
            print(lst_vals)            
            
            if CURPERC == 100:
                NENT = dataDownstream[curcol].nunique()
                possible_combinations = [random.sample(lst_vals,NENT)]
            else:
                possible_combinations = []
                dic_of_strlsts = {}
                
                while True:
                    if len(dic_of_strlsts) == NUM_DIRTY_DT: break
                    tmplst = random.sample(lst_vals, NENT)

                    if str(sorted(tmplst)) not in dic_of_strlsts:
                        dic_of_strlsts[str(sorted(tmplst))] = 1
                        possible_combinations.append(tmplst)     
         
                print('All possible_combinations:')
                print(possible_combinations)  

            print('NENT value is:' + str(NENT))

            noise = PERC_OCCUR
            indexlstlst = []

            bestPerformingModel_LR_full, avgsc_train_lst_LR_full, avgsc_lst_LR_full, avgsc_hld_lst_LR_full = [],[],[],[]
            bestPerformingModel_RF_full, avgsc_train_lst_RF_full, avgsc_lst_RF_full, avgsc_hld_lst_RF_full = [],[],[],[]
            mse_LR_full, bias_LR_full, var_LR_full = [],[],[]
            mse_RF_full, bias_RF_full, var_RF_full = [],[],[]            
            
            k = 0
            for comb in possible_combinations:
                k = k + 1
                print(comb)
                curdataDownstream = copy.deepcopy(dataDownstream)

#                 GRP_SIZE_MINUS_1_lst = [1,1,1,1,1,2,2,2,3,3,4,5]
                GRP_SIZE_MINUS_1_lst = SampleZipf(nr, dxr, NENT)
                print('zipf lst:')
                print(GRP_SIZE_MINUS_1_lst)
                
                for thisiter in range(len(comb)):
                    pq = comb[thisiter]
                    print('\nCurrent value is:' + str(pq))

                    noise = random.uniform(0, 0.50)
                    dirtiness = int(curdic[pq]*noise)
    #                     print(dirtiness)
                    abc = curdataDownstream[curdataDownstream[curcol] == pq].sample(dirtiness, random_state=1)
                    indxlst = abc.index.tolist()
                    indexlstlst.append(indxlst)

                    chk_indxlst = chunkIt(indxlst, GRP_SIZE_MINUS_1_lst[thisiter])

                    for j in range(len(chk_indxlst)): curdataDownstream.loc[chk_indxlst[j], curcol] = str(pq) + '_' + str(j) + '_dummy'    
            
            
                print(curdataDownstream[curcol].value_counts())
                print('\n')
                           

                bestPerformingModel_LR,avgsc_train_lst_LR,avgsc_lst_LR,avgsc_hld_lst_LR = LogRegClassifier(curdataDownstream, y, y_cur, attribute_names, 0)
                bestPerformingModel_RF,avgsc_train_lst_RF,avgsc_lst_RF,avgsc_hld_lst_RF = RandForestClassifier(curdataDownstream, y, y_cur, attribute_names, 0)

                mse_LR, bias_LR, var_LR, mse_RF, bias_RF, var_RF = BiasVarDecomp(curdataDownstream, y, bestPerformingModel_LR, bestPerformingModel_RF)
        
                avgsc_train_lst_LR,avgsc_lst_LR,avgsc_hld_lst_LR = round(avgsc_train_lst_LR*100.0,3), round(avgsc_lst_LR*100.0,3), round(avgsc_hld_lst_LR*100.0,3)
                avgsc_train_lst_RF,avgsc_lst_RF,avgsc_hld_lst_RF = round(avgsc_train_lst_RF*100.0,3), round(avgsc_lst_RF*100.0,3), round(avgsc_hld_lst_RF*100.0,3)
        
                bestPerformingModel_LR_full.append(bestPerformingModel_LR)
                avgsc_train_lst_LR_full.append(avgsc_train_lst_LR)
                avgsc_lst_LR_full.append(avgsc_lst_LR)
                avgsc_hld_lst_LR_full.append(avgsc_hld_lst_LR)

                bestPerformingModel_RF_full.append(bestPerformingModel_RF)
                avgsc_train_lst_RF_full.append(avgsc_train_lst_RF)
                avgsc_lst_RF_full.append(avgsc_lst_RF)
                avgsc_hld_lst_RF_full.append(avgsc_hld_lst_RF)
                
                mse_LR_full.append(mse_LR)
                bias_LR_full.append(bias_LR)
                var_LR_full.append(var_LR)
                
                mse_RF_full.append(mse_RF)
                bias_RF_full.append(bias_RF)
                var_RF_full.append(var_RF)
            
            diff_train_lst_LR_full = [round(scrtr_LR - number,3) for number in avgsc_train_lst_LR_full]
            diff_lst_LR_full = [round(scrval_LR - number,3) for number in avgsc_lst_LR_full] 
            diff_hld_lst_LR_full = [round(scrte_LR - number,3) for number in avgsc_hld_lst_LR_full]
            
            diff_train_lst_RF_full = [round(scrtr_RF - number,3) for number in avgsc_train_lst_RF_full]
            diff_lst_RF_full = [round(scrval_RF - number,3) for number in avgsc_lst_RF_full]
            diff_hld_lst_RF_full = [round(scrte_RF - number,3) for number in avgsc_hld_lst_RF_full]

            diff_mse_LR_full = [(number-mse_LR_TR) for number in mse_LR_full]
            diff_bias_LR_full = [(number-bias_LR_TR) for number in bias_LR_full]
            diff_var_LR_full = [(number-var_LR_TR) for number in var_LR_full]

            diff_mse_RF_full = [(number-mse_RF_TR) for number in mse_RF_full]
            diff_bias_RF_full = [(number-bias_RF_TR) for number in bias_RF_full]
            diff_var_RF_full = [(number-var_RF_TR) for number in var_RF_full]            
            
            sample = open(SAVEPLACE, 'a')
            print('Original Dataset:', file = sample)
            print('---', file = sample)
            print(scrtr_LR, file = sample)
            print(scrval_LR, file = sample)
            print(scrte_LR, file = sample)

            print('---', file = sample)
            print(scrtr_RF, file = sample)
            print(scrval_RF, file = sample)
            print(scrte_RF, file = sample)
#             print('\n', file = sample)            
            
            print('---', file = sample)
            print(dfAblation, file = sample)
            print('\n', file = sample)
            
            print('Original MSE,bias,var -> LR,RF', file = sample)
            print(str(mse_LR_TR) + ' ' + str(bias_LR_TR) + ' ' + str(var_LR_TR), file = sample)
            print(str(mse_RF_TR) + ' ' + str(bias_RF_TR) + ' ' + str(var_RF_TR), file = sample)
            print('\n', file = sample)            
            
            print('Categories that are diluted with duplicates:', file = sample)
            print(possible_combinations[:10], file = sample)

            print('---', file = sample)
            print(avgsc_train_lst_LR_full, file = sample)
            print(avgsc_lst_LR_full, file = sample)
            print(avgsc_hld_lst_LR_full, file = sample)

            print('---', file = sample)
            print(avgsc_train_lst_RF_full, file = sample)
            print(avgsc_lst_RF_full, file = sample)
            print(avgsc_hld_lst_RF_full, file = sample)
            print('\n', file = sample)
            
            print('Difference between Orignal and Duplicate Dataset:', file = sample)
            print('---', file = sample)
            print(diff_train_lst_LR_full, file = sample)
            print(diff_lst_LR_full, file = sample)
            print(diff_hld_lst_LR_full, file = sample)

            print('---', file = sample)
            print(diff_train_lst_RF_full, file = sample)
            print(diff_lst_RF_full, file = sample)
            print(diff_hld_lst_RF_full, file = sample)
            print('\n', file = sample)              
            
            print('MSE,bias,var --- LR', file = sample)
            print(mse_LR_full, file = sample)
            print(bias_LR_full, file = sample)
            print(var_LR_full, file = sample)
            print('\n', file = sample)       
            
            print('MSE,bias,var --- RF', file = sample)
            print(mse_RF_full, file = sample)
            print(bias_RF_full, file = sample)
            print(var_RF_full, file = sample)
            print('\n', file = sample)  
            
            print('Difference MSE,bias,var --- LR', file = sample)
            print(diff_mse_LR_full, file = sample)
            print(diff_bias_LR_full, file = sample)
            print(diff_var_LR_full, file = sample)
            print('\n', file = sample)    
            
            print('Difference MSE,bias,var --- RF', file = sample)
            print(diff_mse_RF_full, file = sample)
            print(diff_bias_RF_full, file = sample)
            print(diff_var_RF_full, file = sample)
            print('\n', file = sample)              
            
            sample.close()
    
    mean_train_acc_LR, mean_val_acc_LR, mean_test_acc_LR = np.mean(avgsc_train_lst_LR_full), np.mean(avgsc_lst_LR_full), np.mean(avgsc_hld_lst_LR_full)
    mean_train_acc_RF, mean_val_acc_RF, mean_test_acc_RF = np.mean(avgsc_train_lst_RF_full), np.mean(avgsc_lst_RF_full), np.mean(avgsc_hld_lst_RF_full)

    return mean_train_acc_LR, mean_val_acc_LR, mean_test_acc_LR, mean_train_acc_RF, mean_val_acc_RF, mean_test_acc_RF    
#     return avgsc_train_lst_LR_full, avgsc_lst_LR_full, avgsc_hld_lst_LR_full, avgsc_train_lst_RF_full, avgsc_lst_RF_full, avgsc_hld_lst_RF_full

def IntroduceDirtinessH2o(dfAblation, dataDownstream, attribute_names, dxr, PERC_OCCUR, scrtr_LR, scrval_LR, scrte_LR, scrtr_RF, scrval_RF, scrte_RF):
    global SAVEPLACE, PERC_ENTY
    categcols = ['duplicateColumn']
    
    ### p1_perc --> Percentage of entities that have duplicates
    ### noise --> Percentage of occurences that are diluted with duplicate values    
    
    GRP_SIZE_MINUS_1 = 1
    PERC_OCCUR = PERC_OCCUR*0.01
    NUM_DIRTY_DT = 10

    for curcol in categcols:
        curdic = dict(dataDownstream[curcol].value_counts())

        p1_perc = [PERC_ENTY]
        for CURPERC in p1_perc:
            print('CURPERC value is:' + str(CURPERC))
            
            tmp = ((dataDownstream[curcol].nunique())*CURPERC)*1.0/100
            NENT = int(round(tmp,0))            

            lst_vals = list(curdic.keys())
            
            print('Dictionary keys:')
            print(lst_vals)            
            
            if CURPERC == 100:
                NENT = dataDownstream[curcol].nunique()
                possible_combinations = [random.sample(lst_vals,NENT)]
            else:
                possible_combinations = []
                dic_of_strlsts = {}
                
                while True:
                    if len(dic_of_strlsts) == NUM_DIRTY_DT: break
                    tmplst = random.sample(lst_vals, NENT)

                    if str(sorted(tmplst)) not in dic_of_strlsts:
                        dic_of_strlsts[str(sorted(tmplst))] = 1
                        possible_combinations.append(tmplst)     
         
                print('All possible_combinations:')
                print(possible_combinations)  
                  

            print('NENT value is:' + str(NENT))

            noise = PERC_OCCUR
            indexlstlst = []

            bestPerformingModel_RF_full, avgsc_train_lst_RF_full, avgsc_lst_RF_full, avgsc_hld_lst_RF_full = [],[],[],[]

            k = 0
            for comb in possible_combinations:
#                 if k > 1: break
                k = k + 1
                print(comb)
                curdataDownstream = copy.deepcopy(dataDownstream)

                for pq in comb:
#                     print('\nCurrent value is:' + str(pq))
    
                    dirtiness = int(curdic[pq]*noise)
#                     print(dirtiness)
                    abc = curdataDownstream[curdataDownstream[curcol] == pq].sample(dirtiness, random_state=1)
                    indxlst = abc.index.tolist()
                    indexlstlst.append(indxlst)

                    chk_indxlst = chunkIt(indxlst, GRP_SIZE_MINUS_1)

                    for j in range(len(chk_indxlst)): curdataDownstream.loc[chk_indxlst[j], curcol] = str(pq) + '_' + str(j) + '_dummy'    

                print(curdataDownstream[curcol].value_counts())
                print('\n')

                bestPerformingModel_RF,avgsc_train_lst_RF,avgsc_lst_RF,avgsc_hld_lst_RF = RandForestH2oClassifier(curdataDownstream, y, y_cur, attribute_names, 'y', 0)
        
                avgsc_train_lst_RF,avgsc_lst_RF,avgsc_hld_lst_RF = round(avgsc_train_lst_RF*100.0,3), round(avgsc_lst_RF*100.0,3), round(avgsc_hld_lst_RF*100.0,3)

                bestPerformingModel_RF_full.append(bestPerformingModel_RF)
                avgsc_train_lst_RF_full.append(avgsc_train_lst_RF)
                avgsc_lst_RF_full.append(avgsc_lst_RF)
                avgsc_hld_lst_RF_full.append(avgsc_hld_lst_RF)

            
            diff_train_lst_RF_full = [round(scrtr_RF - number,3) for number in avgsc_train_lst_RF_full]
            diff_lst_RF_full = [round(scrval_RF - number,3) for number in avgsc_lst_RF_full]
            diff_hld_lst_RF_full = [round(scrte_RF - number,3) for number in avgsc_hld_lst_RF_full]

            sample = open(SAVEPLACE, 'a')
            print('Original Dataset:', file = sample)

            print('---', file = sample)
            print(scrtr_RF, file = sample)
            print(scrval_RF, file = sample)
            print(scrte_RF, file = sample)
#             print('\n', file = sample)            
            
            print('---', file = sample)            
            print(dfAblation, file = sample)
            print('\n', file = sample)             
            
            print('Categories that are diluted with duplicates:', file = sample)
            print(possible_combinations[:10], file = sample)

            print('---', file = sample)
            print(avgsc_train_lst_RF_full, file = sample)
            print(avgsc_lst_RF_full, file = sample)
            print(avgsc_hld_lst_RF_full, file = sample)
            print('\n', file = sample)
            
            print('Difference between Orignal and Duplicate Dataset:', file = sample)

            print('---', file = sample)
            print(diff_train_lst_RF_full, file = sample)
            print(diff_lst_RF_full, file = sample)
            print(diff_hld_lst_RF_full, file = sample)
            print('\n', file = sample)              
            
            sample.close()

    mean_train_acc_RF, mean_val_acc_RF, mean_test_acc_RF = np.mean(avgsc_train_lst_RF_full), np.mean(avgsc_lst_RF_full), np.mean(avgsc_hld_lst_RF_full)

    return mean_train_acc_LR, mean_val_acc_LR, mean_test_acc_LR, mean_train_acc_RF, mean_val_acc_RF, mean_test_acc_RF    
#     return avgsc_train_lst_LR_full, avgsc_lst_LR_full, avgsc_hld_lst_LR_full, avgsc_train_lst_RF_full, avgsc_lst_RF_full, avgsc_hld_lst_RF_full

def mysampler(domsize, nr):
    global dr
    num_per_dom = int(nr/domsize)

    df = pd.DataFrame()
    arr,arr2 = [],[]
    for i in range(domsize):
        for j in range(num_per_dom): 
            arr.append(i)
            arr2.append(np.random.normal(0,1,1).tolist()[0])
            
    curlen = len(arr)
    
    j=0
    while j < (nr-curlen):
        arr.append(j)
        j += 1
             
        
    for k in range(dr):
        random.shuffle(arr)        
        df[k] = arr
    
#     print(df)
        
    rutplst = df.values.tolist()
    return rutplst

def main():
    
    problst = []
    CPTdic = {}

    seed = 100
    random.seed(seed)
    np.random.seed(seed)

    numD = int(sys.argv[1]) # number of clean datasets
    dr = int(sys.argv[2]) # number of categorical features
    dxrlst = [int(sys.argv[3])] # domain size of all categorical features
    trE = int(sys.argv[4]) # number of training examples
    nr = int(5*trE/3) # total number of examples is determined from given trE
    dupColIndex = int(sys.argv[5]) # the duplicate column index, present to 0

    skewPresent = int(sys.argv[6]) # if skew not present in duplication parameters, then only specify the below parameters.
    PERC_ENTY = int(sys.argv[7]) # fraction of entities that are diluted with dups
    PERC_OCCUR = int(sys.argv[8]) # total occurrence value of the duplicate set
    GRP_SIZE_MINUS_1 = int(sys.argv[9]) # duplicate set size
    

    SAVEPLACE = 'logs/synthetic/allx/duplicates-1(dr=4)/Occurrence=' + str(PERC_OCCUR) + '/duplicateColumn(dxr=' + str(dxrlst[0]) + ')(nr=' + str(trE) + ')-' + str(PERC_ENTY) + '.txt'
    sample = open(SAVEPLACE, 'w')

    train_errors_lst, test_errors_lst, train_errors_lst_Categ, test_errors_lst_Categ, train_errors_lst_both, test_errors_lst_both = [],[],[],[],[],[]
    Log_train_errors_lst, Log_test_errors_lst, Log_train_errors_lst_Categ, Log_test_errors_lst_Categ, Log_train_errors_lst_both, Log_test_errors_lst_both = [],[],[],[],[],[]
    val_errors_lst, val_errors_lst_Categ, val_errors_lst_both, Log_val_errors_lst, Log_val_errors_lst_Categ, Log_val_errors_lst_both = [],[],[],[],[],[]
    avg_depth_numeric_lst,avg_depth_categ_lst,avg_depth_both_lst = [],[],[]
    bestPerformingModels_Categ, bestPerformingModels_Numeric, bestPerformingModels_both = [],[],[]
    Log_avgruntime_lst,Log_avgruntime_Categ_lst,Log_avgruntime_both_lst,avgruntime_lst,avgruntime_Categ_lst,avgruntime_both_lst = [],[],[],[],[],[]

    # for KNINT in nonintcategvarslst:
    for dxr in dxrlst:
    #     ProbLst = createCPT(dxr)
        DomainLst = list(range(dxr))
        OHElst = {}
        j=0
        for x in DomainLst:
            lst = [0] * (len(DomainLst))
            lst[j] = 1
            j += 1
            OHElst[x] = lst


        CPTdic = CreateNDCPT(dr,dxr)   
        ########################################################################################################

        avgtestloss, avgtrainloss, avgtestloss_Categ, avgtrainloss_Categ, avgtestloss_both, avgtrainloss_both  = 0.0,0.0,0.0,0.0,0.0,0.0
        Log_avgvalloss , Log_avgvalloss_Categ, Log_avgvalloss_both, avgvalloss, avgvalloss_Categ, avgvalloss_both  = 0.0,0.0,0.0,0.0,0.0,0.0
        Log_avgtestloss , Log_avgtrainloss , Log_avgtestloss_Categ, Log_avgtrainloss_Categ, Log_avgtestloss_both, Log_avgtrainloss_both = 0.0,0.0,0.0,0.0,0.0,0.0
        avg_depth_numeric, avg_depth_categ, avg_depth_both = 0.0,0.0,0.0
        Log_avgruntime,Log_avgruntime_Categ,Log_avgruntime_both,avgruntime,avgruntime_Categ,avgruntime_both = 0.0,0.0,0.0,0.0,0.0,0.0
        first_LR_lst, first_RF_lst, second_LR_lst, second_RF_lst, third_LR_lst, third_RF_lst = [],[],[],[],[],[]

        for ti in range(numD):
            print('START#############################################################################################')

            rtuplst = mysampler(dxr, nr)
            rtuples = {}
            for rid in range(nr):  rtuples[rid] = rtuplst[rid]

        #############################################################################################
        ### Full DataSet
        #############################################################################################    
            fullsetx,fullsetx_Categ,fullsetx_both,fullsety = [],[],[],[]
            for dataid in range(int(nr)):
                fullsetx, fullsetx_Categ, fullsetx_both, fullsety = buildDataset(fullsetx, fullsetx_Categ, fullsetx_both, fullsety, rtuples[dataid],DomainLst,OHElst,dxr)
    #         print(fullsetx_Categ)

            print('-------')        
            print('% 0 Entries:')
            print(fullsety.count(0)*100/len(fullsety))        
            print('-------')        

            dataDownstream, y, y_cur, attribute_names = DatasetToDataFrame(fullsetx_Categ, fullsety)
    #         y_cur = [10,1,1]
            print('#############################################################################################')
            print('Mutual Info:')
            print(sklearn.feature_selection.mutual_info_classif(dataDownstream, y, discrete_features = True))
            print('#############################################################################################')
    #         print(dataDownstream, y, y_cur, attribute_names)
        #############################################################################################
        ### Original LR and RF
        #############################################################################################
            bestPerformingModel_LR,scrtr_LR, scrval_LR, scrte_LR = LogRegClassifier(dataDownstream, y, y_cur, attribute_names, 0)
            bestPerformingModel_RF,scrtr_RF, scrval_RF, scrte_RF = RandForestClassifier(dataDownstream, y, y_cur, attribute_names, 0)


            scrtr_LR, scrval_LR, scrte_LR = round(scrtr_LR*100.0,3), round(scrval_LR*100.0,3), round(scrte_LR*100.0,3)
            scrtr_RF, scrval_RF, scrte_RF = round(scrtr_RF*100.0,3), round(scrval_RF*100.0,3), round(scrte_RF*100.0,3)
            print('#############################################################################################')
            print('Current Datset #:' + str(ti))
            print('Original LR and RF:')
            print(scrtr_LR, scrval_LR, scrte_LR)
            print(scrtr_RF, scrval_RF, scrte_RF)
            print('#############################################################################################')        

            Log_avgtrainloss += scrtr_LR
            Log_avgvalloss += scrval_LR
            Log_avgtestloss += scrte_LR
            Log_avgruntime += 0

            avgtrainloss += scrtr_RF
            avgvalloss += scrval_RF
            avgtestloss += scrte_RF
            avgruntime += 0

            dfAblation,LR_index,RF_index = AblationStudy(dataDownstream, y, y_cur, attribute_names, scrtr_LR, scrval_LR, scrte_LR, scrtr_RF, scrval_RF, scrte_RF)
    #         first_LR_lst.append(LR_index[0]),second_LR_lst.append(LR_index[1]),third_LR_lst.append(LR_index[2])
    #         first_RF_lst.append(RF_index[0]),second_RF_lst.append(RF_index[1]),third_RF_lst.append(RF_index[2])        
            print('#############################################################################################')
            print('Ablation:')
            print(dfAblation)
            print('#############################################################################################')

        ############################################################################################
        ## Duplicates LR and RF
        ############################################################################################
            if skewPresent: mean_train_acc_LR, mean_val_acc_LR, mean_test_acc_LR, mean_train_acc_RF, mean_val_acc_RF, mean_test_acc_RF = IntroduceDirtinessSkew(dfAblation, dataDownstream, attribute_names, dxr, PERC_OCCUR, scrtr_LR, scrval_LR, scrte_LR, scrtr_RF, scrval_RF, scrte_RF)
            else: mean_train_acc_LR, mean_val_acc_LR, mean_test_acc_LR, mean_train_acc_RF, mean_val_acc_RF, mean_test_acc_RF = IntroduceDirtiness(dfAblation, dataDownstream, attribute_names, dxr, PERC_OCCUR, scrtr_LR, scrval_LR, scrte_LR, scrtr_RF, scrval_RF, scrte_RF)

            Log_avgtrainloss_Categ += mean_train_acc_LR
            Log_avgvalloss_Categ += mean_val_acc_LR
            Log_avgtestloss_Categ += mean_test_acc_LR
            Log_avgruntime_Categ += 0

            avgtrainloss_Categ += mean_train_acc_RF
            avgvalloss_Categ += mean_val_acc_RF
            avgtestloss_Categ += mean_test_acc_RF
            avgruntime_Categ += 0        

            print('#############################################################################################')        
            print('Current Datset #:' + str(ti))
            print('Duplicate LR and RF:')

            print(mean_train_acc_LR, mean_val_acc_LR, mean_test_acc_LR)
            print(mean_train_acc_RF, mean_val_acc_RF, mean_test_acc_RF)
            print('#############################################################################################')        

        print('\n Domain size:' + str(dxr))
        print('Logistic Regression:')
        print('Original Accuracy:')
        (Log_avgtrainloss, Log_avgvalloss, Log_avgtestloss, Log_avgruntime) = (float(Log_avgtrainloss)/numD, float(Log_avgvalloss)/numD, float(Log_avgtestloss)/numD, float(Log_avgruntime)/numD)
        print(Log_avgtrainloss, Log_avgvalloss, Log_avgtestloss, Log_avgruntime)

        Log_train_errors_lst.append(Log_avgtrainloss)
        Log_val_errors_lst.append(Log_avgvalloss)
        Log_test_errors_lst.append(Log_avgtestloss)
        Log_avgruntime_lst.append(Log_avgruntime)

        print('Duplicate Accuracy:')
        (Log_avgtrainloss_Categ, Log_avgvalloss_Categ , Log_avgtestloss_Categ, Log_avgruntime_Categ ) = (float(Log_avgtrainloss_Categ)/numD, float(Log_avgvalloss_Categ)/numD, float(Log_avgtestloss_Categ)/numD, float(Log_avgruntime_Categ)/numD)
        print(Log_avgtrainloss_Categ, Log_avgvalloss_Categ , Log_avgtestloss_Categ, Log_avgruntime_Categ)

        Log_train_errors_lst_Categ.append(Log_avgtrainloss_Categ)
        Log_val_errors_lst_Categ.append(Log_avgvalloss_Categ)    
        Log_test_errors_lst_Categ.append(Log_avgtestloss_Categ)    
        Log_avgruntime_Categ_lst.append(Log_avgruntime_Categ)


        print('Random Forest:')
        print('Original Accuracy:')
        (avgtrainloss, avgvalloss, avgtestloss,avg_depth_numeric, avgruntime) = (float(avgtrainloss)/numD, float(avgvalloss)/numD, float(avgtestloss)/numD, float(avg_depth_numeric)/numD, float(avgruntime)/numD)
        print(avgtrainloss, avgvalloss, avgtestloss,avg_depth_numeric, avgruntime)

        train_errors_lst.append(avgtrainloss)
        val_errors_lst.append(avgvalloss)    
        test_errors_lst.append(avgtestloss)
        avg_depth_numeric_lst.append(avg_depth_numeric)
        avgruntime_lst.append(avgruntime)

        print('Duplicate Accuracy:')
        (avgtrainloss_Categ, avgvalloss_Categ , avgtestloss_Categ, avg_depth_categ, avgruntime_Categ ) = (float(avgtrainloss_Categ)/numD, float(avgvalloss_Categ)/numD, float(avgtestloss_Categ)/numD, float(avg_depth_categ)/numD, float(avgruntime_Categ)/numD )
        print(avgtrainloss_Categ, avgvalloss_Categ , avgtestloss_Categ, avg_depth_categ, avgruntime_Categ)

        train_errors_lst_Categ.append(avgtrainloss_Categ)
        val_errors_lst_Categ.append(avgvalloss_Categ)
        test_errors_lst_Categ.append(avgtestloss_Categ)    
        avg_depth_categ_lst.append(avg_depth_categ)
        avgruntime_Categ_lst.append(avgruntime_Categ)

        print('Both Accuracy:')
        (avgtrainloss_both, avgvalloss_both , avgtestloss_both, avg_depth_both, avgruntime_both ) = (float(avgtrainloss_both)/numD, float(avgvalloss_both)/numD, float(avgtestloss_both)/numD, float(avg_depth_both)/numD, float(avgruntime_both)/numD )
        print(avgtrainloss_both, avgvalloss_both , avgtestloss_both, avg_depth_both, avgruntime_both)

        train_errors_lst_both.append(avgtrainloss_both)
        val_errors_lst_both.append(avgvalloss_both)
        test_errors_lst_both.append(avgtestloss_both)    
        avg_depth_both_lst.append(avg_depth_both)    
        avgruntime_both_lst.append(avgruntime_both)

        print('END#############################################################################################')



main()