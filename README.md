<meta name="robots" content="noindex">

# Studying the Impact of Categorical Duplicates on ML

In this work, we take a step towards empirically characterizing the impact of Categorical duplicates on ML classification task with a three-pronged approach. Our benchmark comprises of three components.

(1) *Our Hand-Labeled Datasets.* We create the first large labeled data where true entities within a Categorical column are annotated with duplicate categories.  Our data includes 1262 string Categorical columns from 231 raw CSV files.

(2) *Downstream Benchmark Suite.* Uses 16 real-world datasets to make empirical observations on the effect of Categorical duplicates on five popular classifiers and five encoding mechanisms. 

(3) *Synthetic Study.* Includes Monte Carlo simulation studies to disentangle the impact with different variables impacting ML discretely and better explain the phenomenon.

The tech report for this work is availble [here](https://adalabucsd.github.io/papers/TR_2023_CategDedup.pdf).

## Environment Setup

To run the benchmark, first make sure that the environment is set up and all the packages stated in the `requirements.txt` file are installed. Use the following commands with `Python3.8+`:

```
virtualenv CategDedupBench
source CategDedupBench/bin/activate
pip install -r requirements.txt
```

## Our Labeled Datasets

Please see `Our Labeled Data` to see the sub-directory organization.


## Downstream Benchmark Suite

- Available ML models in our benchmark suite are: 
    - Logisitic Regression (LR)
    - Random Forest (RF)
    - Artificial neural network (ANN)
    - Support Vector Machines (SVM)
    - XGBoost (XGB)

- Available encoding methods in our benchmark suite are:
    - One-hot encoding (OHE)
    - Similarity encoding (SIME)
    - String encoding (STRE)
    - Transformer encoding (TRANSE)
    - TABBIE (TABBIE)

- Our curated set of 16 downstream datasets (Please refer to our [tech report](https://adalabucsd.github.io/papers/TR_2023_CategDedup.pdf) for details into how they were chosen and fine-grained stats on them) are:
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


- Run the downstream benchmark with different configuration settings. Use shorthand notations for model, encoding, data files from above.

1. On specific dataset from our curated set of data files, and a specific model and encoding scheme:
```
python Downstream_Benchmark.py --DataName MIDWEST --EncodingMethod OHE --model LR
```
2. On a data with its path and specific ML model and encoding scheme
```
python Downstream_Benchmark.py --DataName 'datalocation/datafile.csv' --EncodingMethod OHE --model LR
```
3. On specific dataset but across all available ML model and encoding schemes
```
python Downstream_Benchmark.py --DataName MIDWEST
```
4. On specific ML model and encoding scheme but across all available datasets
```
python Downstream_Benchmark.py --EncodingMethod OHE --model LR
```

- Results are written in a directory `Results` in a file titled `DATANAME_results.csv.` The file has the following schema

```
dataset,DuplicationType,encoding,model,lift_acc,overfitting_gap
```

    - `DuplicationType` takes integers corresponding to different duplication types (which we want to study while deduplicating the rest with Truth) presented in Table 3 of the [tech report](https://adalabucsd.github.io/papers/TR_2023_CategDedup.pdf). 
        - 0: Retain all duplication type in Raw data
        - Retain the following types exclusively in the Raw data
            - 1: Capitalization type
            - 2: Misspellings
            - 3: Abbreviation
            - 4: Difference of Special Characters
            - 5: Different Ordering
            - 6: Synonyms
            - 7: Presence of Extra Information
            - 8: Different grammar

    - `encoding` and `model` takes shorthand notations from above

    - `lift_acc` denotes the lift in % diagonal accuracy with Truth relative to Raw data

    - `overfitting_gap` denotes the difference between train and validation accuracies with Truth relative to Raw data

## Synthetic Study

Monte Carlo simulations studying a complex joint distribution where the features obtained the data columns determine the target based on random sampling of conditional probability tables. This is a two step process.

1. Checkpointing. Specify the configuration parameters for the simulation study and log the results. The default directory for writing the log files is set to `logs/`. To understand the influence of each explanatory variable (EV) on ML, we vary one parameter at a time while fixing the rest. Simulations can be run under different setting with the following shell scripts

- Varying the number of training examples in the dataset while the parameters that characterize the amount of duplicates are kept fixed
```
sh varyTrainExamples.sh
```
- Varying the different variables that characterize the magnitude of duplication in a column one at a time with the following scripts
```
sh varyEntities.sh
sh varyOccurrence.sh
sh DuplicateGroupSize.sh
sh RunskewedDuplication.sh
```

2. Visualizations. Specify the parameter to summarize as part of the box and whisker plot. Visualize with the following self-explanatory jupyter notebook.

```
VisualizePlots.ipynb
```


<!-- 2. Hyerplane. A distribution where a true hyperplane separates the classes. -->

<!-- #### 1. Our Labeled Data 

Entities in string categorical columns annotated with duplicates, along with their raw CSV files. 

#### 2. Downstream Benchmark Suite

Downstream datasets with their raw and deduped versions and downstream model source code.

#### 3. Simulation Study

Monte Carlo simulations for AllX and Hyerplane scenario -->