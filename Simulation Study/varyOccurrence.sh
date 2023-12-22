
numD = 100 # number of clean datasets
dr = 3 # number of categorical features
dxr = 10 # domain size of all categorical features

dupColIndex = 0 # the duplicate column index, present to 0

tre = 3000 # number of training examples
skewPresent = 0 # if skew is not present in duplication parameters, then only specify the below parameters.
PercEnty = 30 # fraction of entities that are diluted with duplicates
PercOccur = 25 # total occurrence value of the duplicate set
GrpSize = 1 # duplicate set size

for PercOccur in 10 25 50
    do echo "$numD $dr $dxr $tre $dupColIndex $skewPresent $PercEnty $PercOccur $GrpSize"
    time python RunSimulations.py $numD $dr $dxr $tre $dupColIndex $skewPresent $PercEnty $PercOccur $GrpSize

done