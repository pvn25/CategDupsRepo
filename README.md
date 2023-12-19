<meta name="robots" content="noindex">

### Studying the Impact of Categorical Duplicates on ML

In this work, we take the first step towards empirically characterizing the impact of Categorical duplicates on ML classification with a three-pronged approach. Our benchmark comprises of three components.

(1) *Our Hand-Labeled Datasets.* We create the first large labeled data where true entities within a Categorical column are annotated with duplicate categories.  Our data includes 1262 string Categorical
columns from 231 raw CSV files.

(2) *Downstream Benchmark Suite.* Use 16 real-world datasets to make empirical observations on the effect of Categorical duplicates on five popular classifiers and five encoding mechanisms. 

(3) *Synthetic Study.* Use Monte Carlo simulation studies to disentangle the impact with different variables impacting ML discretely and explain the phenomenon.


## Environment Setup

To set up the environment, first make sure that all the packages stated in the `requirements.txt` file are installed. Use the following commands:

```
virtualenv CategDedupBench
source CategDedupBench/bin/activate
pip install -r requirements.txt
```



<!-- #### 1. Our Labeled Data 

Entities in string categorical columns annotated with duplicates, along with their raw CSV files. 

#### 2. Downstream Benchmark Suite

Downstream datasets with their raw and deduped versions and downstream model source code.

#### 3. Simulation Study

Monte Carlo simulations for AllX and Hyerplane scenario -->