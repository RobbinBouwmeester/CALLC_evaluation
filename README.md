# CALLC_evaluation
Performance and model evaluation of CALLC. The evaluation can be replicated by taking the following steps:

## 1. Generating predictions 

In order to evaluate CALLC we need to generate some predictions. In total there are 5 prediction sets that can be distinguished;

1. Learning curves with duplicate analyte structures across datasets
2. Learning curves without duplicate analyte structures across datasets
3. Cross-validation without duplicate analyte structures across datasets
4. Cross-validation with duplicate analyte structures across datasets
5. Comparison with Aicheler et al

However, we need models in Layer 1, so rerun the following code:

```
CALLC/new_train_l1.py
```

The freshly trained models will be located in:

```
CALLC/mods_l1/
```

With these models we can generate the prediction sets mentioned above. Once the code below from 1.x is run the predictions can be found in:

```
CALLC/test_preds/
```

Place these predictions in the appropiate folder located in:

```
data/predictions/
```

### 1.1 Learning Curve

In order to generate predictions for the learning curve with duplicate analytes structures across datasets, run the following:

```
CALLC/initial_train.py
```

Make sure the main function gets the value 'train/retmetfeatures.csv' for the parameter infilen.

In order to generate predictions for the learning curve without duplicate analytes structures across datasets, run the following:

```
CALLC/initial_train.py
```

Make sure the main function gets the value 'train/retmetfeatures_nodup.csv' for the parameter infilen. 

### 1.2 Cross-Validation

In order to generate predictions for the CV with duplicate analytes structures across datasets, run the following:

```
CALLC/initial_train_CV.py
```

Make sure the main function gets the value 'train/retmetfeatures.csv' for the parameter infilen.

In order to generate predictions for the CV without duplicate analytes structures across datasets, run the following:

```
CALLC/initial_train_CV.py
```

Make sure the main function gets the value 'train/retmetfeatures_nodup.csv' for the parameter infilen. 

### 1.3 Aicheler comparison

In order to generate predictions for the comparison with the aicheler model run the following:

```
CALLC/initial_train_aicheler.py
```

Make sure the main function gets the value 'train/retmetfeatures.csv' for the parameter infilen.

## 2. Parsing predictions 

After you have placed all predictions in:

```
data/predictions/
```

You can run the following script:

```
./parse_predictions.py
```

The predictions will be parsed and for each specific metric a result file will be placed in the appropiate folders here:

```
data/parsed/
```

## 3. Generating figures

Run the following code to replicate the figures:

```
./manuscript_figs.R
```

Make sure all the parsed predictions are located here:

```
data/parsed/
```

Then the figures can be found here:

```
figures/
```
