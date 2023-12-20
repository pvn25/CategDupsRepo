<meta name="robots" content="noindex">

# Studying the Impact of Categorical Duplicates on ML

In this work, we take the first step towards empirically characterizing the impact of Categorical duplicates on ML classification with a three-pronged approach. Our benchmark comprises of three components.

(1) *Our Hand-Labeled Datasets.* We create the first large labeled data where true entities within a Categorical column are annotated with duplicate categories.  Our data includes 1262 string Categorical
columns from 231 raw CSV files.

(2) *Downstream Benchmark Suite.* Use 16 real-world datasets to make empirical observations on the effect of Categorical duplicates on five popular classifiers and five encoding mechanisms. 

(3) *Synthetic Study.* Use Monte Carlo simulation studies to disentangle the impact with different variables impacting ML discretely and explain the phenomenon.


## Environment Setup

To run the benchmark, first make sure that the environment is set up and all the packages stated in the `requirements.txt` file are installed. Use the following commands:

```
virtualenv CategDedupBench
source CategDedupBench/bin/activate
pip install -r requirements.txt
```

## Our Labeled Datasets

Please see `Our Labeled Data` to see the sub-directory organization.


## Downstream Benchmark Suite

Available ML models (with their shorthand used in our code file) in our benchmark suite are: 
- Logisitic Regression (LR)
- Random Forest (RF)
- Artificial neural network (ANN)
- Support Vector Machines (SVM)
- XGBoost (XGB)

Available encoding methods in our benchmark suite are:
- One-hot encoding (OHE)
- Similarity encoding (SIME)
- String encoding (STRE)
- Transformer encoding (TRANSE)
- TABBIE (TABBIE)

Our curated set of 16 downstream datasets (Please refer to our tech report for fine-grained stats on them) are:
- Midwest Survey (MIDWEST)
- Mental Health (MENTAL_HEALTH)
- Relocated Vehicles (RELOC_VEHICLES)
- Health Sciences (HEALTH_SCIENCE)
- Salaries (SALARIES)
- TSM Habitat (TSM)
- EU IT (EUIT)
- Halloween (HALLOWEEN)
- Utility (UTILITY)
- Mid or Feed (MIDFEED)
- Wifi (WIFI)
- Etailing (ETAILING)
- San Francisco (SF)
- Building Violations (BUILDING_VIOLATION)
- US Labor (USLABOR)
- Pet Registration (PETREG)


Run the downstream benchmark with a specific dataset, model, and encoding scheme:
```
python Downstream_Benchmark.py --DataName MIDWEST --EncodingMethod OHE --model LR
```
Run the downstream benchmark with a specific dataset but across all available ML model and encoding schemes
```
python Downstream_Benchmark.py --DataName MIDWEST
```
Run the downstream benchmark with a specific ML model and encoding scheme but across all available datasets
```
python Downstream_Benchmark.py --EncodingMethod OHE --model LR
```
Run the downstream benchmark with a new data path and specific ML model and encoding scheme
```
python Downstream_Benchmark.py --DataName 'datalocation/datafile.csv' --EncodingMethod OHE --model LR
```

<!-- ## Synthetic Study -->


<!-- #### 1. Our Labeled Data 

Entities in string categorical columns annotated with duplicates, along with their raw CSV files. 

#### 2. Downstream Benchmark Suite

Downstream datasets with their raw and deduped versions and downstream model source code.

#### 3. Simulation Study

Monte Carlo simulations for AllX and Hyerplane scenario -->