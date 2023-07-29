from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import tree
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score
from math import sqrt
import numpy as np
import pandas as pd
import sys
from sklearn import metrics
from Featurize_CorrectEncoding import *
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
import time

maxintval = sys.maxsize
Hcurstate = 100

def RandForestH2oClassifier(dataDownstream, y, y_cur,attribute_names,TargetColumn,similarity_flag):
    X_train, X_test,y_train,y_test = train_test_split(dataDownstream,y, test_size=0.2,random_state=Hcurstate)
    X_train_new = X_train.reset_index(drop=True)
    y_train_new = y_train.reset_index(drop=True)
    
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)    

    X_train_new = X_train_new.values
    y_train_new = y_train_new.values

    # print(X_train_new)

    k = 5
    kf = KFold(n_splits=k,random_state=Hcurstate, shuffle=True)
    avg_train_acc,avg_test_acc = 0,0

    n_estimators_grid = [5,25,50,75,100]
    max_depth_grid = [5,10,25,50,100]    

    avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
    avgsc,avgsc_train,avgsc_hld = 0,0,0

    pq = 0
    for train_index, test_index in kf.split(X_train_new):
        X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
        y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]

    #     print(X_train_cur.shape)

        X_train_cur_df, X_test_cur_df, X_test_df = pd.DataFrame(X_train_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test,columns=attribute_names).reset_index(drop=True)
        # print(X_train_cur_df)
        # print('-----')
        if similarity_flag: X_train_cur_df,X_test_cur_df, X_test_df = FeaturizeH2OSimiarity(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)
        else: X_train_cur_df,X_test_cur_df, X_test_df = FeaturizeH2O(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)

        # print(X_train_cur_df)
        
        feature_names, response, dfh_tr, dfh_val, dfh_te, df_try, df_valy, df_tey = Convert2H2oFrame(X_train_cur_df, y_train_cur, X_test_cur_df, y_test_cur, X_test_df, y_test, TargetColumn)
        # print(dfh_tr)
#         print(feature_names)
#         print(response)
        
        bestPerformingModel, scrtr, scrval, scrte, runtime, curdepth = BuildDecisionTreeH2o(feature_names, response, dfh_tr, dfh_val, dfh_te, df_try, df_valy, df_tey)
        
        print('pq:' + str(pq))
        print(scrtr)
        print(scrval)
        print(scrte)
        
        avgsc_train_lst.append(scrtr)
        avgsc_lst.append(scrval)
        avgsc_hld_lst.append(scrte)

        avgsc_train = avgsc_train + scrtr    
        avgsc = avgsc + scrval
        avgsc_hld = avgsc_hld + scrte

        pq += 1
        
    print('5 fold Train, Validation, and Test Accuracies:')
    print(avgsc_train_lst)
    print(avgsc_lst)
    print(avgsc_hld_lst)

    print('Avg Train, Validation, and Test Accuracies:')    
    print(avgsc_train/k)
    print(avgsc/k)
    print(avgsc_hld/k)

    return bestPerformingModel, avgsc_train_lst,avgsc_lst,avgsc_hld_lst


def RandForestH2oClassifier1f(dataDownstream, y, y_cur,attribute_names,TargetColumn,similarity_flag):
    X_train, X_test,y_train,y_test = train_test_split(dataDownstream,y, test_size=0.2,random_state=Hcurstate)
    X_train_new = X_train.reset_index(drop=True)
    y_train_new = y_train.reset_index(drop=True)
    
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)    

    X_train_new = X_train_new.values
    y_train_new = y_train_new.values

    # print(X_train_new)

    k = 5
    kf = KFold(n_splits=k,random_state=Hcurstate, shuffle=True)
    avg_train_acc,avg_test_acc = 0,0

    n_estimators_grid = [5,25,50,75,100]
    max_depth_grid = [5,10,25,50,100]    

    avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
    avgsc,avgsc_train,avgsc_hld = 0,0,0

    pq = 0
    for train_index, test_index in kf.split(X_train_new):
        if pq == 1: break
        pq = pq + 1
        X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
        y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]

    #     print(X_train_cur.shape)

        X_train_cur_df, X_test_cur_df, X_test_df = pd.DataFrame(X_train_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test,columns=attribute_names).reset_index(drop=True)
        # print(X_train_cur_df)
        # print('-----')
        if similarity_flag: X_train_cur_df,X_test_cur_df, X_test_df = FeaturizeH2OSimiarity(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)
        else: X_train_cur_df,X_test_cur_df, X_test_df = FeaturizeH2O(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)

        # print(X_train_cur_df)
        
        feature_names, response, dfh_tr, dfh_val, dfh_te, df_try, df_valy, df_tey = Convert2H2oFrame(X_train_cur_df, y_train_cur, X_test_cur_df, y_test_cur, X_test_df, y_test, TargetColumn)
        # print(dfh_tr)
#         print(feature_names)
#         print(response)
        
        bestPerformingModel, scrtr, scrval, scrte, runtime, curdepth = BuildDecisionTreeH2o(feature_names, response, dfh_tr, dfh_val, dfh_te, df_try, df_valy, df_tey)
        
        print('pq:' + str(pq))
        print(scrtr)
        print(scrval)
        print(scrte)
        
        avgsc_train_lst.append(scrtr)
        avgsc_lst.append(scrval)
        avgsc_hld_lst.append(scrte)

        avgsc_train = avgsc_train + scrtr    
        avgsc = avgsc + scrval
        avgsc_hld = avgsc_hld + scrte

        pq += 1
        
    print('5 fold Train, Validation, and Test Accuracies:')
    print(avgsc_train_lst)
    print(avgsc_lst)
    print(avgsc_hld_lst)

    print('Avg Train, Validation, and Test Accuracies:')    
    print(avgsc_train/k)
    print(avgsc/k)
    print(avgsc_hld/k)

    return bestPerformingModel, avgsc_train_lst,avgsc_lst,avgsc_hld_lst



def Convert2H2oFrame(trainsetx,trainsety,valsetx, valsety,testsetx, testsety, targetCol):
    df_tr,df_val,df_te = pd.DataFrame(trainsetx),pd.DataFrame(valsetx),pd.DataFrame(testsetx)
    df_try,df_valy,df_tey = pd.DataFrame(trainsety,columns=[targetCol]),pd.DataFrame(valsety,columns=[targetCol]),pd.DataFrame(testsety,columns=[targetCol])

#     df_try[targetCol] = df_try[targetCol].apply(lambda x: "'" + str(x) + "'")
#     df_valy[targetCol] = df_valy[targetCol].apply(lambda x: "'" + str(x) + "'")
#     df_tey[targetCol] = df_tey[targetCol].apply(lambda x: "'" + str(x) + "'")

    trainDF,valDF,testDF = pd.concat([df_try,df_tr],axis=1),pd.concat([df_valy,df_val],axis=1),pd.concat([df_tey,df_te],axis=1)
#     print(trainDF)
    response = targetCol

#     for col in trainDF.columns: trainDF.rename(columns={col:"a"+str(col)},inplace=True)
#     for col in valDF.columns: valDF.rename(columns={col:"a"+str(col)},inplace=True)
#     for col in testDF.columns: testDF.rename(columns={col:"a"+str(col)},inplace=True)

    feature_names = trainDF.columns.tolist()

#     print(feature_names)
#     print(trainDF)
    
    dfh_tr,dfh_val,dfh_te = h2o.H2OFrame(trainDF),h2o.H2OFrame(valDF),h2o.H2OFrame(testDF)
    
    return feature_names, response, dfh_tr, dfh_val, dfh_te, df_try, df_valy, df_tey

def BuildDecisionTreeH2o(feature_names, response, dfh_tr, dfh_val, dfh_te, df_try, df_valy, df_tey):
#         print(df_try)
#         print('---')
#         print(df_valy)
#         print('---')        
#         print(df_tey)
#         print('---')
        
        start = time.time()

        n_estimators_grid = [5,25,50,75,100]
        max_depth_grid = [2,3,5,7,10,25,50,100]
        bestPerformingModel = H2ORandomForestEstimator(ntrees=1,max_depth=1,seed=Hcurstate)
        bestscore, bestdepth = 0,0

        for ne in n_estimators_grid:
            for md in max_depth_grid:
                clf = H2ORandomForestEstimator(ntrees=ne,max_depth=md,seed=Hcurstate)
                clf.train(x=feature_names,y=response,training_frame=dfh_tr)

                pred = clf.predict(dfh_val).as_data_frame()['predict'].tolist()
                tru = df_valy[response].tolist()
                sc = accuracy_score(pred,tru)

                if bestscore < sc:
                    bestdepth = md
                    bestscore = sc
                    bestPerformingModel = clf        

        end = time.time()
        runtime = end-start                
                
        pred = bestPerformingModel.predict(dfh_tr).as_data_frame()['predict'].tolist()
        tru = df_try[response].tolist()
        scrtr = accuracy_score(pred,tru)

        pred = bestPerformingModel.predict(dfh_val).as_data_frame()['predict'].tolist()
        tru = df_valy[response].tolist()
        scrval = accuracy_score(pred,tru)

        pred = bestPerformingModel.predict(dfh_te).as_data_frame()['predict'].tolist()
        tru = df_tey[response].tolist()
        scrte = accuracy_score(pred,tru)
        curdepth =  bestPerformingModel.max_depth
        
        print(scrtr, scrval, scrte)
        
        return bestPerformingModel, scrtr, scrval, scrte, runtime, curdepth

def LogRegClassifier(dataDownstream, y, y_cur,attribute_names,similarity_flag):

    X_train, X_test,y_train,y_test = train_test_split(dataDownstream,y, test_size=0.2,random_state=Hcurstate)
    X_train_new = X_train.reset_index(drop=True)
    y_train_new = y_train.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train_new = X_train_new.values
    y_train_new = y_train_new.values

    # print(X_train_new)

    k = 5
    kf = KFold(n_splits=k,random_state=Hcurstate, shuffle=True)
    avg_train_acc,avg_test_acc = 0,0

    val_arr = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

    avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
    avgsc,avgsc_train,avgsc_hld = 0,0,0

    pq = 0
    for train_index, test_index in kf.split(X_train_new):
#         if pq == 1: break
        pq = pq + 1
        X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
        y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]

        # print(X_train_cur)
        # print(X_train_cur.shape)
        # print(attribute_names)

        X_train_cur_df, X_test_cur_df, X_test_df = pd.DataFrame(X_train_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test,columns=attribute_names).reset_index(drop=True)
#         print('train:')
#         print(X_train_cur_df)
#         print('val:')        
#         print(X_test_cur_df)        
#         print('test:')        
#         print(X_test_df)
        
        if similarity_flag: all_cols_train,all_cols_test,all_cols_heldtest = FeaturizeSimiarity(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)
        else: all_cols_train,all_cols_test,all_cols_heldtest = Featurize(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)

#         print('train:')        
        print(all_cols_train.shape)
        # print(all_cols_train)

        bestPerformingModel = LogisticRegression(penalty='l2',C = 1,random_state=Hcurstate)
        bestscore = 0
        for val in val_arr:
            clf = LogisticRegression(penalty='l2',C = val,random_state=Hcurstate)
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

    print('5 fold Train, Validation, and Test Accuracies:')
    print(avgsc_train_lst)
    print(avgsc_lst)
    print(avgsc_hld_lst)

    print('Avg Train, Validation, and Test Accuracies:')    
    print(avgsc_train/k)
    print(avgsc/k)
    print(avgsc_hld/k)

    return bestPerformingModel, avgsc_train_lst,avgsc_lst,avgsc_hld_lst


def LogRegClassifier1f(dataDownstream, y, y_cur,attribute_names,similarity_flag):

    X_train, X_test,y_train,y_test = train_test_split(dataDownstream,y, test_size=0.2,random_state=Hcurstate)
    X_train_new = X_train.reset_index(drop=True)
    y_train_new = y_train.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train_new = X_train_new.values
    y_train_new = y_train_new.values

    # print(X_train_new)

    k = 5
    kf = KFold(n_splits=k,random_state=Hcurstate, shuffle=True)
    avg_train_acc,avg_test_acc = 0,0

    val_arr = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

    avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
    avgsc,avgsc_train,avgsc_hld = 0,0,0

    pq = 0
    for train_index, test_index in kf.split(X_train_new):
        if pq == 1: break
        pq = pq + 1
        X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
        y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]

        # print(X_train_cur)
        # print(X_train_cur.shape)
        # print(attribute_names)

        X_train_cur_df, X_test_cur_df, X_test_df = pd.DataFrame(X_train_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test,columns=attribute_names).reset_index(drop=True)
#         print('train:')
#         print(X_train_cur_df)
#         print('val:')        
#         print(X_test_cur_df)        
#         print('test:')        
#         print(X_test_df)
        
        if similarity_flag: all_cols_train,all_cols_test,all_cols_heldtest = FeaturizeSimiarity(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)
        else: all_cols_train,all_cols_test,all_cols_heldtest = Featurize(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)

#         print('train:')        
        print(all_cols_train.shape)
        # print(all_cols_train)

        bestPerformingModel = LogisticRegression(penalty='l2',C = 1,random_state=Hcurstate)
        bestscore = 0
        for val in val_arr:
            clf = LogisticRegression(penalty='l2',C = val,random_state=Hcurstate)
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

    print('5 fold Train, Validation, and Test Accuracies:')
    print(avgsc_train_lst)
    print(avgsc_lst)
    print(avgsc_hld_lst)

    print('Avg Train, Validation, and Test Accuracies:')    
    print(avgsc_train/k)
    print(avgsc/k)
    print(avgsc_hld/k)

    return bestPerformingModel, avgsc_train_lst,avgsc_lst,avgsc_hld_lst    
    
def RandForestClassifier(dataDownstream, y, y_cur,attribute_names,similarity_flag):
    X_train, X_test,y_train,y_test = train_test_split(dataDownstream,y, test_size=0.2,random_state=Hcurstate)
    X_train_new = X_train.reset_index(drop=True)
    y_train_new = y_train.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train_new = X_train_new.values
    y_train_new = y_train_new.values

    # print(X_train_new)

    k = 5
    kf = KFold(n_splits=k,random_state=Hcurstate, shuffle=True)
    avg_train_acc,avg_test_acc = 0,0

    n_estimators_grid = [5,25,50,75,100]
    max_depth_grid = [5,10,25,50,100]    

    avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
    avgsc,avgsc_train,avgsc_hld = 0,0,0

    pq = 0
    for train_index, test_index in kf.split(X_train_new):
#         if pq == 1: break
        pq = pq + 1
        
        X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
        y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]

    #     print(X_train_cur.shape)

        X_train_cur_df, X_test_cur_df, X_test_df = pd.DataFrame(X_train_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test,columns=attribute_names).reset_index(drop=True)
#         print(X_train_cur_df)
        
        if similarity_flag: all_cols_train,all_cols_test,all_cols_heldtest = FeaturizeSimiarity(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)
        else: all_cols_train,all_cols_test,all_cols_heldtest = Featurize(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)

        print(all_cols_train.shape)
        # print(all_cols_train)

        bestPerformingModel = RandomForestClassifier(n_estimators=10,max_depth=5, random_state=Hcurstate)
        bestscore = 0
        for ne in n_estimators_grid:
            for md in max_depth_grid:
                clf = RandomForestClassifier(n_estimators=ne,max_depth=md, random_state=Hcurstate)
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

    print('5 fold Train, Validation, and Test Accuracies:')
    print(avgsc_train_lst)
    print(avgsc_lst)
    print(avgsc_hld_lst)

    print('Avg Train, Validation, and Test Accuracies:')    
    print(avgsc_train/k)
    print(avgsc/k)
    print(avgsc_hld/k)

    return bestPerformingModel, avgsc_train_lst,avgsc_lst,avgsc_hld_lst


def RandForestClassifier1f(dataDownstream, y, y_cur,attribute_names,similarity_flag):
    X_train, X_test,y_train,y_test = train_test_split(dataDownstream,y, test_size=0.2,random_state=Hcurstate)
    X_train_new = X_train.reset_index(drop=True)
    y_train_new = y_train.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train_new = X_train_new.values
    y_train_new = y_train_new.values

    # print(X_train_new)

    k = 5
    kf = KFold(n_splits=k,random_state=Hcurstate, shuffle=True)
    avg_train_acc,avg_test_acc = 0,0

    n_estimators_grid = [5,25,50,75,100]
    max_depth_grid = [5,10,25,50,100]    

    avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
    avgsc,avgsc_train,avgsc_hld = 0,0,0

    pq = 0
    for train_index, test_index in kf.split(X_train_new):
        if pq == 1: break
        pq = pq + 1
        
        X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
        y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]

    #     print(X_train_cur.shape)

        X_train_cur_df, X_test_cur_df, X_test_df = pd.DataFrame(X_train_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test,columns=attribute_names).reset_index(drop=True)
#         print(X_train_cur_df)
        
        if similarity_flag: all_cols_train,all_cols_test,all_cols_heldtest = FeaturizeSimiarity(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)
        else: all_cols_train,all_cols_test,all_cols_heldtest = Featurize(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)

        print(all_cols_train.shape)
        # print(all_cols_train)

        bestPerformingModel = RandomForestClassifier(n_estimators=10,max_depth=5, random_state=Hcurstate)
        bestscore = 0
        for ne in n_estimators_grid:
            for md in max_depth_grid:
                clf = RandomForestClassifier(n_estimators=ne,max_depth=md, random_state=Hcurstate)
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

    print('5 fold Train, Validation, and Test Accuracies:')
    print(avgsc_train_lst)
    print(avgsc_lst)
    print(avgsc_hld_lst)

    print('Avg Train, Validation, and Test Accuracies:')    
    print(avgsc_train/k)
    print(avgsc/k)
    print(avgsc_hld/k)

    return bestPerformingModel, avgsc_train_lst,avgsc_lst,avgsc_hld_lst



def LinearRegression(dataDownstream, y, y_cur,attribute_names,similarity_flag):

    X_train, X_test,y_train,y_test = train_test_split(dataDownstream,y, test_size=0.2,random_state=Hcurstate)   
    
    X_train_new = X_train.reset_index(drop=True)    
    y_train_new = y_train.reset_index(drop=True)    
    
    X_test = X_test.reset_index(drop=True)  
    y_test = y_test.reset_index(drop=True)  
    
    X_train_new = X_train_new.values    
    y_train_new = y_train_new.values

    k = 5
    kf = KFold(n_splits=k,random_state=Hcurstate)
    avg_train_acc,avg_test_acc = 0,0

    val_arr = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

    avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
    avgsc,avgsc_train,avgsc_hld = 0,0,0

    i=0
    for train_index, test_index in kf.split(X_train_new):
#         if i>0: break
#         i=i+1
        X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
        y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]


        X_train_cur_df, X_test_cur_df, X_test_df = pd.DataFrame(X_train_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test,columns=attribute_names).reset_index(drop=True) 
        print('train:') 
        print(X_train_cur_df)   
#         print('val:')         
#         print(X_test_cur_df)          
#         print('test:')            
#         print(X_test_df)  
            
        if similarity_flag: all_cols_train,all_cols_test,all_cols_heldtest = FeaturizeSimiarity(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)   
        else: all_cols_train,all_cols_test,all_cols_heldtest = Featurize(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)

        print(all_cols_train.shape)
        
        bestPerformingModel = Ridge(alpha=1.0,random_state=Hcurstate)
        bestscore = maxintval

        for val in val_arr:
            clf = Ridge(alpha=val,random_state=Hcurstate)
            clf = clf.fit(all_cols_train, y_train_cur)
            y_pred = clf.predict(all_cols_test)
            sc = sqrt(mean_squared_error(y_pred, y_test_cur))
#             print(sc)
            if bestscore > sc:
                bestscore = sc
                bestPerformingModel = clf


        y_pred = bestPerformingModel.predict(all_cols_train)
        bscr_train = sqrt(mean_squared_error(y_pred, y_train_cur))
        
        y_pred = bestPerformingModel.predict(all_cols_test)
        bscr = sqrt(mean_squared_error(y_pred, y_test_cur))
        
        y_pred = bestPerformingModel.predict(all_cols_heldtest)
        bscr_hld = sqrt(mean_squared_error(y_pred, y_test))

        avgsc_train_lst.append(bscr_train)
        avgsc_lst.append(bscr)
        avgsc_hld_lst.append(bscr_hld)

        avgsc_train = avgsc_train + bscr_train    
        avgsc = avgsc + bscr
        avgsc_hld = avgsc_hld + bscr_hld

        print(bscr_train)
        print(bscr)
        print(bscr_hld)
    

    print('5-fold Train, Validation, and Test loss:')
    print(avgsc_train_lst)
    print(avgsc_lst)
    print(avgsc_hld_lst)
    
    print('Avg Train, Validation, and Test loss:')    
    print(avgsc_train/k)
    print(avgsc/k)
    print(avgsc_hld/k)
    
    return bestPerformingModel, avgsc_train_lst,avgsc_lst,avgsc_hld_lst

def RandForestRegressor(dataDownstream, y, y_cur,attribute_names,similarity_flag):
    X_train, X_test,y_train,y_test = train_test_split(dataDownstream,y, test_size=0.2,random_state=Hcurstate)

    X_train_new = X_train.reset_index(drop=True)
    y_train_new = y_train.reset_index(drop=True)
    
    X_test = X_test.reset_index(drop=True)  
    y_test = y_test.reset_index(drop=True)

    X_train_new = X_train_new.values
    y_train_new = y_train_new.values

    k = 5
    kf = KFold(n_splits=k,random_state=Hcurstate)
    avg_train_acc,avg_test_acc = 0,0

    n_estimators_grid = [5,25,50,75,100]
    max_depth_grid = [5,10,25,50,100]

    avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
    avgsc,avgsc_train,avgsc_hld = 0,0,0

    i=0
    for train_index, test_index in kf.split(X_train_new):
#         if i>0: break
#         i=i+1        
        X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
        y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]

        X_train_cur_df, X_test_cur_df, X_test_df = pd.DataFrame(X_train_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test,columns=attribute_names).reset_index(drop=True) 
#         print(X_train_cur_df) 
            
        if similarity_flag: all_cols_train,all_cols_test,all_cols_heldtest = FeaturizeSimiarity(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)   
        else: all_cols_train,all_cols_test,all_cols_heldtest = Featurize(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)  
        print(all_cols_train.shape) 
        # print(all_cols_train)
        
        bestPerformingModel = RandomForestRegressor(n_estimators=10,max_depth=5, random_state=Hcurstate)
        bestscore = maxintval
        for ne in n_estimators_grid:
            for md in max_depth_grid:
                clf = RandomForestRegressor(n_estimators=ne,max_depth=md, random_state=Hcurstate)
                clf = clf.fit(all_cols_train, y_train_cur)
                
                y_pred = clf.predict(all_cols_test)
                sc = sqrt(mean_squared_error(y_pred, y_test_cur))                
                # sc = clf.score(all_cols_test, y_test_cur)

                if bestscore > sc:
                    bestscore = sc
                    bestPerformingModel = clf

        y_pred = bestPerformingModel.predict(all_cols_train)
        bscr_train = sqrt(mean_squared_error(y_pred, y_train_cur))
        
        y_pred = bestPerformingModel.predict(all_cols_test)
        bscr = sqrt(mean_squared_error(y_pred, y_test_cur))
        
        y_pred = bestPerformingModel.predict(all_cols_heldtest)
        bscr_hld = sqrt(mean_squared_error(y_pred, y_test))

        avgsc_train_lst.append(bscr_train)
        avgsc_lst.append(bscr)
        avgsc_hld_lst.append(bscr_hld)

        avgsc_train = avgsc_train + bscr_train    
        avgsc = avgsc + bscr
        avgsc_hld = avgsc_hld + bscr_hld

        print(bscr_train)
        print(bscr)
        print(bscr_hld)
    
    print('5-fold Train, Validation, and Test loss:')
    print(avgsc_train_lst)
    print(avgsc_lst)
    print(avgsc_hld_lst)
    
    print('Avg Train, Validation, and Test loss:')    
    print(avgsc_train/k)
    print(avgsc/k)
    print(avgsc_hld/k)

    # y_pred = bestPerformingModel.predict(X_test)
    return bestPerformingModel, avgsc_train_lst,avgsc_lst,avgsc_hld_lst


def MLPRegressorr(data1,y):
    
    X_train, X_test,y_train,y_test = train_test_split(data1,y, test_size=0.2,random_state=Hcurstate)

    X_train_new = X_train.reset_index(drop=True)
    y_train_new = y_train.reset_index(drop=True)
    
    X_train_new = X_train_new.values
    y_train_new = y_train_new.values

    k = 5
    kf = KFold(n_splits=k,random_state=Hcurstate)
    avg_train_acc,avg_test_acc = 0,0

    n_estimators_grid = [5,25,50,75,100]
    max_depth_grid = [5,10,25,50,100]

    avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
    avgsc,avgsc_train,avgsc_hld = 0,0,0

    i=0
    for train_index, test_index in kf.split(X_train_new):
#         if i>0: break
#         i=i+1        
        X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
        y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]
        X_train_train, X_val,y_train_train,y_val = train_test_split(X_train_cur,y_train_cur, test_size=0.25,random_state=Hcurstate)

        print(X_train_train.shape)
        print(X_val.shape)            
        
        bestPerformingModel = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=300 , random_state=Hcurstate)
        bestPerformingModel = bestPerformingModel.fit(X_train, y_train)
        print(bestPerformingModel.n_layers_)

        y_pred = bestPerformingModel.predict(X_train_cur)
        bscr_train = sqrt(mean_squared_error(y_pred, y_train_cur))
        
        y_pred = bestPerformingModel.predict(X_test_cur)
        bscr = sqrt(mean_squared_error(y_pred, y_test_cur))
        
        y_pred = bestPerformingModel.predict(X_test)
        bscr_hld = sqrt(mean_squared_error(y_pred, y_test))

        avgsc_train_lst.append(bscr_train)
        avgsc_lst.append(bscr)
        avgsc_hld_lst.append(bscr_hld)

        avgsc_train = avgsc_train + bscr_train    
        avgsc = avgsc + bscr
        avgsc_hld = avgsc_hld + bscr_hld

        print(bscr_train)
        print(bscr)
        print(bscr_hld)
    
    print('5-fold Train, Validation, and Test loss:')
    print(avgsc_train_lst)
    print(avgsc_lst)
    print(avgsc_hld_lst)
    
    print('Avg Train, Validation, and Test loss:')    
    print(avgsc_train/k)
    print(avgsc/k)
    print(avgsc_hld/k)

    y_pred = bestPerformingModel.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    return avgsc_train_lst,avgsc_lst,avgsc_hld_lst


# def MLPClassifierr(data1,y):
    
#     X_train, X_test,y_train,y_test = train_test_split(data1,y, test_size=0.2,random_state=Hcurstate)

#     X_train_new = X_train.reset_index(drop=True)
#     y_train_new = y_train.reset_index(drop=True)
    
#     X_train_new = X_train_new.values
#     y_train_new = y_train_new.values

#     k = 5
#     kf = KFold(n_splits=k,random_state=Hcurstate)
#     avg_train_acc,avg_test_acc = 0,0

#     reg_grid = [0.0001,0.001,0.01]

#     avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
#     avgsc,avgsc_train,avgsc_hld = 0,0,0

#     i=0
#     for train_index, test_index in kf.split(X_train_new):
# #         if i>0: break
# #         i=i+1        
#         X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
#         y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]
#         X_train_train, X_val,y_train_train,y_val = train_test_split(X_train_cur,y_train_cur, test_size=0.25,random_state=Hcurstate)

#         print(X_train_train.shape)
#         print(X_val.shape)            
        
#         bestPerformingModel = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=100 , random_state=Hcurstate)
#         bestPerformingModel = bestPerformingModel.fit(X_train, y_train)

#         bscr_train = bestPerformingModel.score(X_train_cur, y_train_cur)
#         bscr = bestPerformingModel.score(X_test_cur, y_test_cur)
#         bscr_hld = bestPerformingModel.score(X_test, y_test)

#         avgsc_train_lst.append(bscr_train)
#         avgsc_lst.append(bscr)
#         avgsc_hld_lst.append(bscr_hld)

#         avgsc_train = avgsc_train + bscr_train    
#         avgsc = avgsc + bscr
#         avgsc_hld = avgsc_hld + bscr_hld

#         print(bscr_train)
#         print(bscr)
#         print(bscr_hld)
    
#     print('5-fold Train, Validation, and Test Accuracies:')
#     print(avgsc_train_lst)
#     print(avgsc_lst)
#     print(avgsc_hld_lst)
    
#     print('Avg Train, Validation, and Test Accuracies:')    
#     print(avgsc_train/k)
#     print(avgsc/k)
#     print(avgsc_hld/k)

#     y_pred = bestPerformingModel.predict(X_test)
#     cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

#     return avgsc_train_lst,avgsc_lst,avgsc_hld_lst


def MLPClassifierr(dataDownstream, y, y_cur,attribute_names,similarity_flag):
    X_train, X_test,y_train,y_test = train_test_split(dataDownstream,y, test_size=0.2,random_state=Hcurstate)
    X_train_new = X_train.reset_index(drop=True)
    y_train_new = y_train.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train_new = X_train_new.values
    y_train_new = y_train_new.values

    # print(X_train_new)

    k = 5
    kf = KFold(n_splits=k,random_state=Hcurstate)
    avg_train_acc,avg_test_acc = 0,0 

    avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
    avgsc,avgsc_train,avgsc_hld = 0,0,0
    full_metrics_micro_lst,full_metrics_macro_lst=[],[]
    full_metrics_micro_val_lst,full_metrics_macro_val_lst=[],[]

    reg_grid = [0.0001,0.001,0.01]

    pq = 0
    for train_index, test_index in kf.split(X_train_new):
#         if pq == 1: break
        pq = pq + 1
        
        X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
        y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]

    #     print(X_train_cur.shape)

        X_train_cur_df, X_test_cur_df, X_test_df = pd.DataFrame(X_train_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test_cur,columns=attribute_names).reset_index(drop=True),pd.DataFrame(X_test,columns=attribute_names).reset_index(drop=True)
#         print(X_train_cur_df)
        
        if similarity_flag: all_cols_train,all_cols_test,all_cols_heldtest = FeaturizeSimiarity(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)
        else: all_cols_train,all_cols_test,all_cols_heldtest = Featurize(X_train_cur_df,X_test_cur_df,X_test_df,attribute_names,y_cur)

        print(all_cols_train.shape)

        bestPerformingModel = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=100 , random_state=Hcurstate)
        # bestPerformingModel = bestPerformingModel.fit(all_cols_train, y_train_cur)      

        bestscore = 0
        for val in reg_grid:
            clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=100 , random_state=Hcurstate, alpha = val)
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

    print('5 fold Train, Validation, and Test Accuracies:')
    print(avgsc_train_lst)
    print(avgsc_lst)
    print(avgsc_hld_lst)

    print('Avg Train, Validation, and Test Accuracies:')    
    print(avgsc_train/k)
    print(avgsc/k)
    print(avgsc_hld/k)

    # print(full_metrics_micro_lst)
    # print(full_metrics_macro_lst)

    return bestPerformingModel, avgsc_train_lst,avgsc_lst,avgsc_hld_lst

