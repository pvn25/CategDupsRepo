from Load_Predictions import *
from downstream_model_CorrectEncoding import *
from Featurize_CorrectEncoding import *
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
import copy
import warnings
import random, re
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import copy
import math
import scipy
import time
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence

warnings.filterwarnings("ignore")
random.seed(100)

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

embedding = TransformerDocumentEmbeddings('roberta-base')
stops = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Use EncodingMethodFlag = 0 with One-hot
# Use EncodingMethodFlag = 1 with Similarity
# Use EncodingMethodFlag = 2 with BERT
EncodingMethodFlag = 0
# To Standardize Column with NLTK Set NormalizeCol flag to 1
NormalizeCol = 1
# Set DataName from 'MIDWEST', 'MENTAL_HEALTH', 'RELOC_VEHICLES', 'HEALTH_SCIENCE', 'SALARIES', 'TSM' , 'EUIT', 'HALLOWEEN', 'UTILITY', 'MIDFEED', 'WIFI', 'ETAILING', 'SF', 'BUILDING_VIOLATION'
DataName = 'MIDWEST'


if DataName == 'MIDWEST':
    InputDownPath = 'Midwest_Survey/In your own words, what would you call the part of the country you live in now_.csv'
    InputFilePath = 'Midwest_Survey/Midwest_Survey.csv'
    TargetColumn = 'Location (Census Region)'
    duplicateColumn = 'In your own words, what would you call the part of the country you live in now?'
elif DataName == 'MENTAL_HEALTH':
    InputFilePath = 'Mental-Health/survey_2014.csv'
    duplicateColumn = 'Gender'
    TargetColumn = 'How easy is it for you to take medical leave for a mental health condition?'
    InputDownPath = 'Mental-Health/Gender.csv'
elif DataName == 'RELOC_VEHICLES':
    InputFilePath = 'Relocated Vehicles/Relocated_Vehicles.csv'
    duplicateColumn = 'Relocated To Street Name'
    TargetColumn = 'Relocated To Direction'
    InputDownPath = 'Relocated Vehicles/Relocated To Street Name.csv'
elif DataName == 'HEALTH_SCIENCE':
    InputFilePath = 'Health-Sciences/RawData_HealthScience.csv'
    duplicateColumn = 'What school(s) do you work in?'
    TargetColumn = 'Does your lab/research group currently use a naming convention to save your data files? '  
    InputDownPath = 'Health-Sciences/What school(s) do you work in_.csv'  
elif DataName == 'SALARIES':
    InputFilePath = 'Salaries/salaries_clean.csv'
    duplicateColumn = 'location_name'
    TargetColumn = 'job_title_category'  
    InputDownPath = 'Salaries/location_name.csv'  
elif DataName == 'TSM':
    InputFilePath = 'TSM Habitat/TSM_Habitat_Rapid_Assessment_Survey.csv'
    duplicateColumn = 'Species'
    TargetColumn = 'Stratum'   
    InputDownPath = 'TSM Habitat/Species.csv' 
elif DataName == 'EUIT':
    InputFilePath = 'IT Salary EU/IT Salary Survey EU  2020.csv'
    duplicateColumn = 'Your main technology / programming language'
    InputDownPath = 'IT Salary EU/Your main technology programming language.csv'
    TargetColumn = 'Position '    
elif DataName == 'HALLOWEEN':
    InputFilePath = 'Halloween/halloween_raw.csv'
    duplicateColumn = 'What.was.your.favorite.candy.'
    TargetColumn = 'What.is.your.age.group.'
    InputDownPath = 'Halloween/What.was.your.favorite.candy..csv'    
elif DataName == 'UTILITY':
    InputFilePath = 'Utility/utility-company-customer-service-response-index-csri-beginning-2005.csv'
    duplicateColumn = 'Service Provider'
    TargetColumn = 'CSRI'
    InputDownPath = 'Utility/Service Provider.csv'
elif DataName == 'MIDFEED':
    InputFilePath = 'Mid or Feed/rawscores.csv'
    duplicateColumn = 'gender'
    TargetColumn = 'B6'    
    InputDownPath = 'Mid or Feed/gender.csv'
elif DataName == 'WIFI':
    InputFilePath = 'Wifi/WiFiSurvey.csv'
    duplicateColumn = 'Locations'
    TargetColumn = 'TechCenter'    
    InputDownPath = 'Wifi/Locations.csv'
elif DataName == 'ETAILING':
    InputFilePath = 'Etailing/etailing.csv'
    duplicateColumn = 'State'
    TargetColumn = 'What is the maximum cart value you ever shopped?'    
    InputDownPath = 'Etailing/State.csv'
elif DataName ==  'SF':
    InputFilePath = 'San Francisco/SalariesInSanFranciscoDataSet.csv'
    duplicateColumn = 'JobTitle'
    TargetColumn = 'TotalPay'   
    InputDownPath = 'San Francisco/JobTitle.csv' 
elif DataName == 'BUILDING_VIOLATION':
    InputFilePath = 'Violations/Violations.csv'
    duplicateColumn = 'Entity or Person(s)'
    TargetColumn = 'Disposition Description'    
    InputDownPath = 'Violations/Entity or Person(s).csv'


def complex_function(x): 
    x = x.lower().strip()
    x = re.sub('[^A-Za-z0-9 ]+', ' ', x)
    x = re.sub("\s\s+" , " ", x)
    x = lemmatizer.lemmatize(x)    
    x = ' '.join([word for word in x.split(' ') if word not in stops])    
    return x

def RunAllModels(dataDownstream,y,y_cur,attribute_names,EncodingMethodFlag):

    lrm,avgsc_train_lst_LR,avgsc_lst_LR,avgsc_hld_lst_LR,test_tuple_LR,val_tuple_LR,train_tuple_LR = LogRegClassifier(dataDownstream,y,y_cur,attribute_names,EncodingMethodFlag)
    print()
    print(val_tuple_LR)
    print(test_tuple_LR)
    # CalculatePRF(full_metrics_micro_lst_LR,full_metrics_macro_lst_LR)

    print()
    rfm,avgsc_train_lst_RF,avgsc_lst_RF,avgsc_hld_lst_RF,test_tuple_RF,val_tuple_RF,train_tuple_RF = RandForestClassifier(dataDownstream,y,y_cur,attribute_names,EncodingMethodFlag)
    print()
    print(val_tuple_RF)
    print(test_tuple_RF)
    # CalculatePRF(full_metrics_micro_lst_RF,full_metrics_macro_lst_RF)

    print()
    mlp,avgsc_train_lst_MLP,avgsc_lst_MLP,avgsc_hld_lst_MLP,test_tuple_MLP,val_tuple_MLP,train_tuple_MLP = MLPClassifierr(dataDownstream,y,y_cur,attribute_names,EncodingMethodFlag)
    print()
    print(val_tuple_MLP)
    print(test_tuple_MLP)
    # CalculatePRF(full_metrics_micro_lst_MLP,full_metrics_macro_lst_MLP)

    portno = 0
    h2o.init(port=54321+portno)
    h2o.no_progress()
    portno=portno+1
    dataDownstream_h2o = copy.deepcopy(dataDownstream)
    rfm_h2o, avgsc_train_lst_h2o,avgsc_lst_h2o,avgsc_hld_lst_h2o,test_tuple_h2o,val_tuple_h2o,train_tuple_h2o = RandForestH2oClassifier(dataDownstream_h2o, y, y_cur,attribute_names,TargetColumn,EncodingMethodFlag)
    print()
    print(val_tuple_h2o)
    print(test_tuple_h2o)
    # CalculatePRF(full_metrics_micro_lst_h20,full_metrics_macro_lst_h20)


def DeduplicateData(dataDownstream):
    df = pd.read_csv(InputDownPath)
    df = df.sort_values(by=['times_entered'], ascending=False)
    df = df.sort_values(by=['group'])
    df = df.reset_index()

    prvgrpno = 0
    dicdups = {}
    curdup = ''
    for index, row in df.iterrows():
        curgrpno = row['group']
        if prvgrpno != curgrpno: 
            curdup = row[duplicateColumn]
            dicdups[curdup] = curdup
        else: dicdups[row[duplicateColumn]] = curdup    
        prvgrpno = curgrpno

    dataDownstream_dedup = copy.deepcopy(dataDownstream)
    dataDownstream_dedup[duplicateColumn] = dataDownstream_dedup[duplicateColumn].fillna('0')
    def func(x):
        if x == '0': return '0'
        return dicdups[x]
        
    dataDownstream_dedup[duplicateColumn] = dataDownstream_dedup[duplicateColumn].apply(lambda x: func(x))
    attribute_names = dataDownstream_dedup.columns.values.tolist()

    dupcol_lst_values = dataDownstream_dedup[duplicateColumn].values.tolist()
    embed_lst = []
    for word in dupcol_lst_values:
        sentence = Sentence(word)
        embedding.embed(sentence)
        tmp_tensor = sentence.embedding
        tmp_lst = tmp_tensor.tolist()
        embed_lst.append(tmp_lst)
        
    arr = np.array(embed_lst)
    df = pd.DataFrame(arr)
    dataDownstream_dedup = pd.concat([dataDownstream_dedup,df], axis=1, sort=False)

    return dataDownstream_dedup



file = InputFilePath
target_col = TargetColumn

dataDownstream = pd.read_csv(file)
dataDownstream = dataDownstream.sample(frac=1, random_state=100)
dataDownstream = dataDownstream[dataDownstream[TargetColumn].notna()]
dataDownstream = dataDownstream[dataDownstream[duplicateColumn].notna()]
dataDownstream = dataDownstream.fillna('0')
dataDownstream = dataDownstream.reset_index(drop=True)
y = dataDownstream[[target_col]]
dataDownstream = dataDownstream.drop(target_col, axis=1)


if EncodingMethodFlag == 2:
    dupcol_lst_values = dataDownstream[duplicateColumn].values.tolist()
    embed_lst = []
    for word in dupcol_lst_values:
        sentence = Sentence(word)
        embedding.embed(sentence)
        tmp_tensor = sentence.embedding
        tmp_lst = tmp_tensor.tolist()
        embed_lst.append(tmp_lst)
        
    arr = np.array(embed_lst)
    df = pd.DataFrame(arr)
    dataDownstream = pd.concat([dataDownstream,df], axis=1, sort=False)

dataFeaturized = FeaturizeFile(dataDownstream)

if DataName == 'MIDWEST':
    curlst = ['RespondentID', 'In your own words, what would you call the part of the country you live in now?', 'Personally identification as a Midwesterner?', 'Illinois in MW?', 'Indiana in MW?', 'Iowa in MW?', 'Kansas in MW?', 'Michigan in MW?', 'Minnesota in MW?', 'Missouri in MW?', 'Nebraska in MW?', 'North Dakota in MW?', 'Ohio in MW?', 'South Dakota in MW?', 'Wisconsin in MW?', 'Arkansas in MW?', 'Colorado in MW?', 'Kentucky in MW?', 'Oklahoma in MW?', 'Pennsylvania in MW?', 'West Virginia in MW?', 'Montana in MW?', 'Wyoming in MW?', 'ZIP Code', 'Gender', 'Age', 'Household Income', 'Education']
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}
    for x in attribute_names:
        if x == 'RespondentID' or x == 'ZIP Code': attribute_dic[x] = 7
        elif x == duplicateColumn: 
            if EncodingMethodFlag == 2: attribute_dic[x] = 7
            else: attribute_dic[x] = 10
        elif str(x).isdigit(): attribute_dic[x] = 0
        else: attribute_dic[x] = 1
elif DataName == 'MENTAL_HEALTH':
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}
    for x in attribute_names:
        if x == 'Age': attribute_dic[x] = 0
        elif x == 'Timestamp': attribute_dic[x] = 7
        elif x == duplicateColumn:
            if EncodingMethodFlag: attribute_dic[x] = 7
            else: attribute_dic[x] = 10        
        elif str(x).isdigit(): attribute_dic[x] = 0
        else: attribute_dic[x] = 1
elif DataName == 'RELOC_VEHICLES':
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}
    for x in attribute_names:
        if x in ['Relocated Date','Plate', 'Relocated From Address Number','Relocated To Address Number','Relocated To Suffix', 'Service Request Number', 'Relocated From X Coordinate', 'Relocated From Y Coordinate', 'Relocated From Latitude', 'Relocated From Longitude', 'Relocated From Location']: attribute_dic[x] = 7
        elif x == duplicateColumn: attribute_dic[x] = 7
        elif str(x).isdigit(): attribute_dic[x] = 0
        else: attribute_dic[x] = 1
elif DataName == 'HEALTH_SCIENCE':
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}
    for x in attribute_names:
        if x in [ 'You selected ÔøΩYesÔøΩ in the question: ÔøΩDo you take additional precautions when you store or share sensitive information?ÔøΩ Please list how in the space below.' ,    'You selected ÔøΩYesÔøΩ in the question: ÔøΩHave you searched for publically available raw data to use in your research?ÔøΩ  Please list the resources/databases you have searched in the space below.']: attribute_dic[x] = 7
        elif x in ['Please describe a data management issue you have experienced in the past two years.  Such a problem might be investigators not willing to share data, lost data due to server crash, difficulty finding data due to inconsistent file naming, lack of understanding of Data Management Plan requirements, etcÔøΩ ']: attribute_dic[x] = 3
        elif x == duplicateColumn: attribute_dic[x] = 7
        elif str(x).isdigit(): attribute_dic[x] = 0
        else: attribute_dic[x] = 1
    print(attribute_dic)    
elif DataName == 'SALARIES':
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}

    labelsfile = pd.read_csv('Salaries/dataFeaturized-salaries.csv')
    for index,row in labelsfile.iterrows(): attribute_dic[row['Attribute_name']] = row['label']

    for x in attribute_names:
        if x == duplicateColumn: attribute_dic[x] = 7
        elif str(x).isdigit(): attribute_dic[x] = 0
elif DataName == 'TSM':
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}

    labelsfile = pd.read_csv('TSM Habitat/dataFeaturized-tsm-habitat.csv')
    for index,row in labelsfile.iterrows(): attribute_dic[row['Attribute_name']] = row['label']

    for x in attribute_names:
        if x == duplicateColumn: attribute_dic[x] = 7
        elif str(x).isdigit(): attribute_dic[x] = 0    
elif DataName == 'EUIT':
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}

    labelsfile = pd.read_csv('IT Salary EU/dataFeaturized-it-salary-eu.csv')
    for index,row in labelsfile.iterrows(): attribute_dic[row['Attribute_name']] = row['label']

    for x in attribute_names:
        if x == duplicateColumn: attribute_dic[x] = 7
        elif str(x).isdigit(): attribute_dic[x] = 0    
elif DataName == 'HALLOWEEN':
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}

    labelsfile = pd.read_csv('Halloween/dataFeaturized-halloween.csv')
    for index,row in labelsfile.iterrows(): attribute_dic[row['Attribute_name']] = row['label']

    for x in attribute_names:
        if x == duplicateColumn: attribute_dic[x] = 7
        elif str(x).isdigit(): attribute_dic[x] = 0    
elif DataName == 'UTILITY':
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}
    labelsfile = pd.read_csv('Utility/dataFeaturized-utility.csv')
    for index,row in labelsfile.iterrows(): attribute_dic[row['Attribute_name']] = row['label']
    attribute_dic['CRM Index'] = 7;attribute_dic['ERM Index'] = 7;attribute_dic['PCM Index'] = 7;attribute_dic['CSM Index'] = 7;attribute_dic['CSRI'] = 7
    for x in attribute_names:
        if x == duplicateColumn: attribute_dic[x] = 7
        elif str(x).isdigit(): attribute_dic[x] = 0
elif DataName == 'MIDFEED':
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}
    labelsfile = pd.read_csv('Mid or Feed/dataFeaturized-mid-feed.csv')
    for index,row in labelsfile.iterrows(): attribute_dic[row['Attribute_name']] = row['label']
    for x in attribute_names:
        if x == duplicateColumn: attribute_dic[x] = 7
        elif str(x).isdigit(): attribute_dic[x] = 0
elif DataName == 'WIFI':
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}

    labelsfile = pd.read_csv('Wifi/dataFeaturized-wifi.csv')
    for index,row in labelsfile.iterrows(): attribute_dic[row['Attribute_name']] = row['label']

    for x in attribute_names:
        if x == duplicateColumn: attribute_dic[x] = 7
        elif str(x).isdigit(): attribute_dic[x] = 0
elif DataName == 'ETAILING':
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}

    labelsfile = pd.read_csv('Etailing/dataFeaturized-etailing.csv')
    for index,row in labelsfile.iterrows(): attribute_dic[row['Attribute_name']] = row['label']

    for x in attribute_names:
        if x == duplicateColumn: attribute_dic[x] = 7
        elif str(x).isdigit(): attribute_dic[x] = 0    
elif DataName ==  'SF':
    attribute_names = dataDownstream.columns.values.tolist()
    attribute_dic = {}
    labelsfile = pd.read_csv('San Francisco/dataFeaturized-sf.csv')
    for index,row in labelsfile.iterrows(): attribute_dic[row['Attribute_name']] = row['label']

    for x in attribute_names:
        if x == duplicateColumn: attribute_dic[x] = 7
        elif str(x).isdigit(): attribute_dic[x] = 0    
elif DataName == 'BUILDING_VIOLATION':
    attribute_names = dataDownstream.columns.values.tolist()
    print(attribute_names)
    attribute_dic = {}

    labelsfile = pd.read_csv('Violations/dataFeaturized-violations.csv')
    for index,row in labelsfile.iterrows(): attribute_dic[row['Attribute_name']] = row['label']

    for x in attribute_names:
        if x == duplicateColumn: attribute_dic[x] = 7
        elif str(x).isdigit(): attribute_dic[x] = 0



attribute_names = dataDownstream.columns.values.tolist()
y_cur = []
for x in attribute_names: y_cur.append(attribute_dic[x])

if NormalizeCol == 1:
    dataDownstream[duplicateColumn] = dataDownstream[duplicateColumn].apply(complex_function)


# Results with Raw Data
RunAllModels(dataDownstream,y,y_cur,attribute_names,EncodingMethodFlag)


# Results with Deduplicated Data
dataDownstream = DeduplicateData(dataDownstream)
RunAllModels(dataDownstream,y,y_cur,attribute_names,EncodingMethodFlag)
