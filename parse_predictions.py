"""
Robbin Bouwmeester

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
This code is used to train retention time predictors and store
predictions from a CV procedure for further analysis.

This project was made possible by MASSTRPLAN. MASSTRPLAN received funding 
from the Marie Sklodowska-Curie EU Framework for Research and Innovation 
Horizon 2020, under Grant Agreement No. 675132.
"""

__author__ = "Robbin Bouwmeester"
__credits__ = ["Robbin Bouwmeester","Prof. Lennart Martens","Prof. Sven Degroeve"]
__license__ = "Apache License, Version 2.0"
__version__ = "1.0"
__maintainer__ = "Robbin Bouwmeester"
__email__ = "robbin.bouwmeester@ugent.be"

import os
from os import listdir
from os.path import isfile, join
import pandas
from scipy.stats import pearsonr
import numpy

l1_scores = {}
l2_scores = {}
l3_scores = {}
l4_scores = {}
own_scores = {}
own_scores_svm = {}
own_scores_bay = {}
own_scores_adab = {}
own_scores_lass = {}

all_preds_l1 = []
all_preds_l2 = []
all_preds_l3 = []

def get_l1_cols(names):
    """
    Extract column names related to Layer 1 
    
    Parameters
    ----------
    names : list
        list with column names
    
    Returns
    -------
    list
        list containing only Layer 1 column names
    """
	ret_names = []
	for n in names:
		if "RtGAM" in n: continue
		if n in ["IDENTIFIER", "time", "preds", "IDENTIFIER.1", "time.1"]: continue
		ret_names.append(n)
	return(ret_names)

def get_l2_cols(names):
	"""
    Extract column names related to Layer 2 
    
    Parameters
    ----------
    names : list
        list with column names
    
    Returns
    -------
    list
        list containing only Layer 2 column names
    """
	ret_names = []
	for n in names:
		if "RtGAMSE" in n: continue 
		if "RtGAM" in n: ret_names.append(n)
	return(ret_names)

def select_pred_files(model_fn,curr_ana_l):
	"""
    Get all files related to a particular analysis
    
    Parameters
    ----------
    model_fn : list
        all files in a particular analysis folder
    curr_ana_l : list
        name of model we are going to analyze
    
    Returns
    -------
    list
		files that are required for a particular analysis
    """
	files_ana = []
	for f in model_fn:
		f_split = f.replace(".csv","").split("_")
		analyze = True

		for temp_c in curr_ana_l: 
			if temp_c not in f_split: analyze = False
		if analyze: 
			if "preds" in f_split and len(f_split) - len(curr_ana_l) == 1:
				files_ana.append(f)
			if "train" in f_split and len(f_split) - len(curr_ana_l) == 2:
				files_ana.append(f)

			
	return(files_ana)

def get_df(files_ana,dir_df="./Data/Predictions/duplicates/"):
	"""
    Get the dataframe associated with this analysis
    
    Parameters
    ----------
    files_ana : list
        list with files that need to be included in the analysis
    dir_df : str
        location of the files
    
    Returns
    -------
    pd.DataFrame
        dataframe related to the training set
	pd.DataFrame
		dataframe related to the testing set
	str
		number of training molecules
	str
		repetition number
	str
		experiment name
    """
	for f in files_ana:
			print("Analyzing file:",join(dir_df, f))
			try:
				df = pandas.read_csv(join(dir_df, f))
			except:
				print("Could not read: %s" % (f))
				continue
			if "train" in f: df_train = df
			else: df_test = df

			if "_train_" not in f:
				num_train = f.split("_")[-2]
				num_rep = f.split("_")[-1].split(".")[0]
				experiment = "_".join(f.split("_")[0:-2])
	return(df_train,df_test,num_train,num_rep,experiment)

def get_cor(l1_cols_train,l2_cols_train,own_cols_xgb,own_cols_svm,own_cols_bay,own_cols_adab,own_cols_lass,df_train,df_test,experiment,fold_num=0):
	"""
    Use correlation as an evaluation metric and extract the appropiate columns to calculate the metric
    
    Parameters
    ----------
    l1_cols_train : list
        list with names for the Layer 1 training columns
	l2_cols_train : list
        list with names for the Layer 2 training columns
	own_cols_xgb : list
        list with names for the Layer 1 xgb columns
	own_cols_svm : list
        list with names for the Layer 1 svm columns
	own_cols_bay : list
        list with names for the Layer 1 brr columns
	own_cols_adab : list
        list with names for the Layer 1 adaboost columns
	own_cols_lass : list
        list with names for the Layer 1 lasso columns
	df_train : pd.DataFrame
        dataframe for training predictions
	df_test : pd.DataFrame
        dataframe for testing predictions
	experiment : str
        dataset name
	fold_num : int
		number for the fold
    
    Returns
    -------
	float
		best correlation for Layer 1
	float
		best correlation for Layer 2
	float
		best correlation for Layer 3
	float
		best correlation for all layers
	float
		correlation for xgb
	float
		correlation for svm
	float
		correlation for brr
	float
		correlation for adaboost
	float
		correlation for lasso
	list
		selected predictions Layer 2
	list
		error for the selected predictions Layer 2
	float
		train correlation for Layer 3
    """
	# Get the pearson values for the layers	
	l1_scores = [pearsonr(df_train[c],df_train["time"])[0] for c in l1_cols_train]
	l2_scores = [pearsonr(df_train[c],df_train["time"])[0] for c in l2_cols_train]

	own_scores_xgb = [pearsonr(df_train[c],df_train["time"])[0] for c in own_cols_xgb]
	own_scores_svm = [pearsonr(df_train[c],df_train["time"])[0] for c in own_cols_svm]
	own_scores_bay = [pearsonr(df_train[c],df_train["time"])[0] for c in own_cols_bay]
	own_scores_lass = [pearsonr(df_train[c],df_train["time"])[0] for c in own_cols_adab]
	own_scores_adab = [pearsonr(df_train[c],df_train["time"])[0] for c in own_cols_lass]

	own_scores_l2 = [x/float(len(df_train["time"])) for x in list(df_train[l2_cols_train].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(sum,axis="rows"))]

	selected_col_l1 = l1_cols_train[l1_scores.index(max(l1_scores))]

	selected_col_l2 = l2_cols_train[l2_scores.index(max(l2_scores))]

	# Reset to 0.0 pearson if we cannot extract the column
	try: selected_col_own_xgb = own_cols_xgb[own_scores_xgb.index(max(own_scores_xgb))]
	except: selected_col_own_xgb = 0.0
	try: selected_col_own_svm = own_cols_svm[own_scores_svm.index(max(own_scores_svm))]
	except: selected_col_own_svm = 0.0
	try: selected_col_own_bay = own_cols_bay[own_scores_bay.index(max(own_scores_bay))]
	except: selected_col_own_bay = 0.0
	try: selected_col_own_lass = own_cols_lass[own_scores_lass.index(max(own_scores_lass))]
	except: selected_col_own_lass = 0.0
	try: selected_col_own_adab = own_cols_adab[own_scores_adab.index(max(own_scores_adab))]
	except: selected_col_own_adab = 0.0

	# Get the best Layer performance
	cor_l1 = pearsonr(df_test["time"],df_test[selected_col_l1])[0]
	cor_l2 = pearsonr(df_test["time"],df_test[selected_col_l2])[0]

	# Reset to 0.0 pearson if we cannot extract the column
	try: cor_own_xgb = pearsonr(df_test["time"],df_test[selected_col_own_xgb])[0]
	except: cor_own_xgb = 0.0
	try: cor_own_svm = pearsonr(df_test["time"],df_test[selected_col_own_svm])[0]
	except: cor_own_svm = 0.0
	try: cor_own_bay = pearsonr(df_test["time"],df_test[selected_col_own_bay])[0]
	except: cor_own_bay = 0.0
	try: cor_own_lass = pearsonr(df_test["time"],df_test[selected_col_own_lass])[0]
	except: cor_own_lass = 0.0
	try: cor_own_adab = pearsonr(df_test["time"],df_test[selected_col_own_adab])[0]
	except: cor_own_adab = 0.0

	cor_l3 = pearsonr(df_test["time"],df_test["preds"])[0]

	# Variables holding all predictions across experiments
	all_preds_l1.extend(zip(df_test["time"],df_test[selected_col_l1],[experiment]*len(df_test[selected_col_l1]),[len(df_train.index)]*len(df_test[selected_col_l1]),[fold_num]*len(df_test[selected_col_l1]),df_test[selected_col_own_xgb],df_test[selected_col_own_bay],df_test[selected_col_own_lass],df_test[selected_col_own_adab]))
	all_preds_l2.extend(zip(df_test["time"],df_test[selected_col_l2],[experiment]*len(df_test[selected_col_l2]),[len(df_train.index)]*len(df_test[selected_col_l2]),[fold_num]*len(df_test[selected_col_l2])))
	all_preds_l3.extend(zip(df_test["time"],df_test["preds"],[experiment]*len(df_test["preds"]),[len(df_train.index)]*len(df_test["preds"]),[fold_num]*len(df_test["preds"])))

	# Also get the training correlation
	train_cor_l1 = pearsonr(df_train["time"],df_train[selected_col_l1])[0]
	train_cor_l2 = pearsonr(df_train["time"],df_train[selected_col_l2])[0]
	train_cor_l3 = pearsonr(df_train["time"],df_train["preds"])[0]

	print()
	print("Error l1: %s,%s" % (train_cor_l1,cor_l1))
	print("Error l2: %s,%s" % (train_cor_l2,cor_l2))
	print("Error l3: %s,%s" % (train_cor_l3,cor_l3))
	print(selected_col_l1,selected_col_l2,selected_col_own_xgb)
	print()
	print()
	print("-------------")

	# Try to select the best Layer, this becomes Layer 4
	cor_l4 = 0.0
	if (train_cor_l1 < train_cor_l2) and (train_cor_l1 < train_cor_l3): cor_l4 = cor_l1
	elif (train_cor_l2 < train_cor_l1) and (train_cor_l2 < train_cor_l3): cor_l4 = cor_l2
	else: cor_l4 = cor_l3

	return(cor_l1,cor_l2,cor_l3,cor_l4,cor_own_xgb,cor_own_svm,cor_own_bay,cor_own_adab,cor_own_lass,list(df_test[selected_col_l2]),list(df_test["time"]-df_test[selected_col_l2]),train_cor_l3) #.replace("RtGAM","RtGAMSE")


def get_avgerr(l1_cols_train,l2_cols_train,own_cols_xgb,own_cols_svm,own_cols_bay,own_cols_adab,own_cols_lass,df_train,df_test,experiment,fold_num=0):
	"""
    Use mae as an evaluation metric and extract the appropiate columns to calculate the metric
    
    Parameters
    ----------
    l1_cols_train : list
        list with names for the Layer 1 training columns
	l2_cols_train : list
        list with names for the Layer 2 training columns
	own_cols_xgb : list
        list with names for the Layer 1 xgb columns
	own_cols_svm : list
        list with names for the Layer 1 svm columns
	own_cols_bay : list
        list with names for the Layer 1 brr columns
	own_cols_adab : list
        list with names for the Layer 1 adaboost columns
	own_cols_lass : list
        list with names for the Layer 1 lasso columns
	df_train : pd.DataFrame
        dataframe for training predictions
	df_test : pd.DataFrame
        dataframe for testing predictions
	experiment : str
        dataset name
	fold_num : int
		number for the fold
    
    Returns
    -------
	float
		best mae for Layer 1
	float
		best mae for Layer 2
	float
		best mae for Layer 3
	float
		best mae for all layers
	float
		mae for xgb
	float
		mae for svm
	float
		mae for brr
	float
		mae for adaboost
	float
		mae for lasso
	list
		selected predictions Layer 2
	list
		error for the selected predictions Layer 2
	float
		train mae for Layer 3
    """
	# Get the mae
	l1_scores = [x/float(len(df_train["time"])) for x in list(df_train[l1_cols_train].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(sum,axis="rows"))]
	l2_scores = [x/float(len(df_train["time"])) for x in list(df_train[l2_cols_train].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(sum,axis="rows"))]
	own_scores_xgb = [x/float(len(df_train["time"])) for x in list(df_train[own_cols_xgb].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(sum,axis="rows"))]
	own_scores_svm = [x/float(len(df_train["time"])) for x in list(df_train[own_cols_svm].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(sum,axis="rows"))]
	own_scores_bay = [x/float(len(df_train["time"])) for x in list(df_train[own_cols_bay].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(sum,axis="rows"))]
	own_scores_lass = [x/float(len(df_train["time"])) for x in list(df_train[own_cols_lass].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(sum,axis="rows"))]
	own_scores_adab = [x/float(len(df_train["time"])) for x in list(df_train[own_cols_adab].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(sum,axis="rows"))]

	own_scores_l2 = [x/float(len(df_train["time"])) for x in list(df_train[l2_cols_train].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(sum,axis="rows"))]

	selected_col_l1 = l1_cols_train[l1_scores.index(min(l1_scores))]
	selected_col_l2 = l2_cols_train[l2_scores.index(min(l2_scores))]

	# Set mae to 0.0 if not able to get column
	try: selected_col_own_xgb = own_cols_xgb[own_scores_xgb.index(min(own_scores_xgb))]
	except KeyError: selected_col_own_xgb = 0.0
	try: selected_col_own_svm = own_cols_svm[own_scores_svm.index(min(own_scores_svm))]
	except KeyError: selected_col_own_svm = 0.0
	try: selected_col_own_bay = own_cols_bay[own_scores_bay.index(min(own_scores_bay))]
	except KeyError: selected_col_own_bay = 0.0
	try: selected_col_own_lass = own_cols_lass[own_scores_lass.index(min(own_scores_lass))]
	except KeyError: selected_col_own_lass = 0.0
	try: selected_col_own_adab = own_cols_adab[own_scores_adab.index(min(own_scores_adab))]
	except KeyError: selected_col_own_adab = 0.0

	# Remove problems with seemingly duplicate columns getting selected
	try:
		cor_l1 = sum(map(abs,df_test["time"]-df_test[selected_col_l1]))/len(df_test["time"])
	except KeyError:
		selected_col_l1 = selected_col_l1.split(".")[0]
		cor_l1 = sum(map(abs,df_test["time"]-df_test[selected_col_l1]))/len(df_test["time"])
	try:
		cor_l2 = sum(map(abs,df_test["time"]-df_test[selected_col_l2]))/len(df_test["time"])
	except KeyError:
		selected_col_l2 = selected_col_l2.split(".")[0]
		cor_l2 = sum(map(abs,df_test["time"]-df_test[selected_col_l2]))/len(df_test["time"])
	try: 
		cor_own_xgb = sum(map(abs,df_test["time"]-df_test[selected_col_own_xgb]))/len(df_test["time"])
	except KeyError:
		selected_col_own_xgb = selected_col_own_xgb.split(".")[0]
		cor_own_xgb = sum(map(abs,df_test["time"]-df_test[selected_col_own_xgb]))/len(df_test["time"])
	try:
		cor_own_svm = sum(map(abs,df_test["time"]-df_test[selected_col_own_svm]))/len(df_test["time"])
	except KeyError:
		selected_col_own_svm = selected_col_own_svm.split(".")[0]
		cor_own_svm = sum(map(abs,df_test["time"]-df_test[selected_col_own_svm]))/len(df_test["time"])
	try:
		cor_own_bay = sum(map(abs,df_test["time"]-df_test[selected_col_own_bay]))/len(df_test["time"])
	except KeyError:
		selected_col_own_bay = selected_col_own_bay.split(".")[0]
		cor_own_bay = sum(map(abs,df_test["time"]-df_test[selected_col_own_bay]))/len(df_test["time"])
	try:
		cor_own_lass = sum(map(abs,df_test["time"]-df_test[selected_col_own_lass]))/len(df_test["time"])
	except KeyError:
		selected_col_own_lass = selected_col_own_lass.split(".")[0]
		cor_own_lass = sum(map(abs,df_test["time"]-df_test[selected_col_own_lass]))/len(df_test["time"])
	try:
		cor_own_adab = sum(map(abs,df_test["time"]-df_test[selected_col_own_adab]))/len(df_test["time"])
	except KeyError:
		selected_col_own_adab = selected_col_own_adab.split(".")[0]
		cor_own_adab = sum(map(abs,df_test["time"]-df_test[selected_col_own_adab]))/len(df_test["time"])

	cor_l3 = sum(map(abs,df_test["time"]-df_test["preds"]))/len(df_test["time"])

	# Variables holding all predictions across experiments
	all_preds_l1.extend(zip(df_test["time"],df_test[selected_col_l1],[experiment]*len(df_test[selected_col_l1]),[len(df_train.index)]*len(df_test[selected_col_l1]),[fold_num]*len(df_test[selected_col_l1]),df_test[selected_col_own_xgb],df_test[selected_col_own_bay],df_test[selected_col_own_lass],df_test[selected_col_own_adab]))
	all_preds_l2.extend(zip(df_test["time"],df_test[selected_col_l2],[experiment]*len(df_test[selected_col_l2]),[len(df_train.index)]*len(df_test[selected_col_l2]),[fold_num]*len(df_test[selected_col_l2])))
	all_preds_l3.extend(zip(df_test["time"],df_test["preds"],[experiment]*len(df_test["preds"]),[len(df_train.index)]*len(df_test["preds"]),[fold_num]*len(df_test["preds"])))

	# Also get the mae for the training models
	train_cor_l1 = sum(map(abs,df_train["time"]-df_train[selected_col_l1]))/len(df_train["time"])
	train_cor_l2 = sum(map(abs,df_train["time"]-df_train[selected_col_l2]))/len(df_train["time"])
	train_cor_l3 = sum(map(abs,df_train["time"]-df_train["preds"]))/len(df_train["time"])

	print()
	print("Error l1: %s,%s" % (train_cor_l1,cor_l1))
	print("Error l2: %s,%s" % (train_cor_l2,cor_l2))
	print("Error l3: %s,%s" % (train_cor_l3,cor_l3))
	print(selected_col_l1,selected_col_l2,selected_col_own_xgb)
	print()
	print()
	print("-------------")

	# Try to select the best Layer, this becomes Layer 4
	cor_l4 = 0.0
	if (train_cor_l1 < train_cor_l2) and (train_cor_l1 < train_cor_l3): cor_l4 = cor_l1
	elif (train_cor_l2 < train_cor_l1) and (train_cor_l2 < train_cor_l3): cor_l4 = cor_l2
	else: cor_l4 = cor_l3

	return(cor_l1,cor_l2,cor_l3,cor_l4,cor_own_xgb,cor_own_svm,cor_own_bay,cor_own_adab,cor_own_lass,list(df_test[selected_col_l2]),list(df_test["time"]-df_test[selected_col_l2]),train_cor_l3)


def get_sumabserr(l1_cols_train,l2_cols_train,df_train,df_test):
	"""
    Use sumabserr as an evaluation metric and extract the appropiate columns to calculate the metric
    
    Parameters
    ----------
    l1_cols_train : list
        list with names for the Layer 1 training columns
	l2_cols_train : list
        list with names for the Layer 2 training columns
	df_train : pd.DataFrame
        dataframe for training predictions
	df_test : pd.DataFrame
        dataframe for testing predictions
    
    Returns
    -------
	float
		best mae for Layer 1
	float
		best mae for Layer 2
	float
		best mae for Layer 3
	float
		best mae for all layers
    """
	l1_scores = list(df_train[l1_cols_train].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(sum,axis="rows"))
	l2_scores = list(df_train[l2_cols_train].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(sum,axis="rows"))

	selected_col_l1 = l1_cols_train[l1_scores.index(min(l1_scores))]
	selected_col_l2 = l2_cols_train[l2_scores.index(min(l2_scores))]

	cor_l1 = sum(map(abs,df_test["time"]-df_test[selected_col_l1]))
	cor_l2 = sum(map(abs,df_test["time"]-df_test[selected_col_l2]))
	cor_l3 = sum(map(abs,df_test["time"]-df_test["preds"]))

	train_cor_l1 = sum(map(abs,df_train["time"]-df_train[selected_col_l1]))
	train_cor_l2 = sum(map(abs,df_train["time"]-df_train[selected_col_l2]))
	train_cor_l3 = sum(map(abs,df_train["time"]-df_train["preds"]))

	cor_l4 = 0.0
	if (train_cor_l1 < train_cor_l2) and (train_cor_l1 < train_cor_l3): cor_l4 = cor_l1
	elif (train_cor_l2 < train_cor_l1) and (train_cor_l2 < train_cor_l3): cor_l4 = cor_l2
	else: cor_l4 = cor_l3

	return(cor_l1,cor_l2,cor_l3,cor_l4)

def get_abserr(l1_cols_train,l2_cols_train,own_cols,df_train,df_test):
	"""
    Use the median error as an evaluation metric and extract the appropiate columns to calculate the metric
    
    Parameters
    ----------
    l1_cols_train : list
        list with names for the Layer 1 training columns
	l2_cols_train : list
        list with names for the Layer 2 training columns
	df_train : pd.DataFrame
        dataframe for training predictions
	df_test : pd.DataFrame
        dataframe for testing predictions
    
    Returns
    -------
	float
		best median error for Layer 1
	float
		best median error for Layer 2
	float
		best median error for Layer 3
	float
		best median error for all layers
	float
		best median error for only models trained in its own dataset
    """
	l1_scores = list(df_train[l1_cols_train].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(numpy.median,axis="rows"))
	l2_scores = list(df_train[l2_cols_train].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(numpy.median,axis="rows"))
	own_scores = list(df_train[own_cols].sub(df_train["time"].squeeze(),axis=0).apply(abs).apply(numpy.median,axis="rows"))

	selected_col_l1 = l1_cols_train[l1_scores.index(min(l1_scores))]
	selected_col_l2 = l2_cols_train[l2_scores.index(min(l2_scores))]
	selected_col_own = own_cols[own_scores.index(min(own_scores))]

	cor_l1 = numpy.median(map(abs,df_test["time"]-df_test[selected_col_l1]))
	cor_l2 = numpy.median(map(abs,df_test["time"]-df_test[selected_col_l2]))
	cor_own = numpy.median(map(abs,df_test["time"]-df_test[selected_col_own]))
	cor_l3 = numpy.median(map(abs,df_test["time"]-df_test["preds"]))

	train_cor_l1 = numpy.median(map(abs,df_train["time"]-df_train[selected_col_l1]))
	train_cor_l2 = numpy.median(map(abs,df_train["time"]-df_train[selected_col_l2]))
	train_cor_l3 = numpy.median(map(abs,df_train["time"]-df_train["preds"]))

	cor_l4 = 0.0
	if (train_cor_l1 < train_cor_l2) and (train_cor_l1 < train_cor_l3): cor_l4 = cor_l1
	elif (train_cor_l2 < train_cor_l1) and (train_cor_l2 < train_cor_l3): cor_l4 = cor_l2
	else: cor_l4 = cor_l3

	return(cor_l1,cor_l2,cor_l3,cor_l4,cor_own)


def saves_scores(cor_l1,cor_l2,cor_l3,cor_l4,cor_own,cor_own_svm,cor_own_bay,cor_own_adab,cor_own_lass,num_train):
	"""
    Save the score to a global variable, prepare to write
    
    Parameters
    ----------
    cor_l1 : float
		best mae for Layer 1
	cor_l2 : float
		best mae for Layer 2
	cor_l3 : float
		best mae for Layer 3
	cor_l4 : float
		best mae for all layers
	cor_own : float
		mae for xgb
	cor_own_svm : float
		mae for svm
	cor_own_bay : float
		mae for brr
	cor_own_adab : float
		mae for adaboost
	cor_own_lass : float
		mae for lasso
	num_train : float
		number of calibration analytes
    
    Returns
    -------

    """
	if num_train in l1_scores.keys(): l1_scores[num_train].append(cor_l1)
	else: l1_scores[num_train] = [cor_l1]
	if num_train in l2_scores.keys(): l2_scores[num_train].append(cor_l2)
	else: l2_scores[num_train] = [cor_l2]
	if num_train in l3_scores.keys(): l3_scores[num_train].append(cor_l3)
	else: l3_scores[num_train] = [cor_l3]
	if num_train in l4_scores.keys(): l4_scores[num_train].append(cor_l4)
	else: l4_scores[num_train] = [cor_l4]
	if num_train in own_scores.keys(): own_scores[num_train].append(cor_own)
	else: own_scores[num_train] = [cor_own]

	if num_train in own_scores_svm.keys(): own_scores_svm[num_train].append(cor_own_svm)
	else: own_scores_svm[num_train] = [cor_own_svm]
	if num_train in own_scores_bay.keys(): own_scores_bay[num_train].append(cor_own_bay)
	else: own_scores_bay[num_train] = [cor_own_bay]
	if num_train in own_scores_adab.keys(): own_scores_adab[num_train].append(cor_own_adab)
	else: own_scores_adab[num_train] = [cor_own_adab]
	if num_train in own_scores_lass.keys(): own_scores_lass[num_train].append(cor_own_lass)
	else: own_scores_lass[num_train] = [cor_own_lass]
	

def get_preds_layers(experiments,model_fn,select="avgerr",dir_df="./Data/Predictions/duplicates/"):
	"""
    Do the main analysis for a specific dataset
    
    Parameters
    ----------
    experiments : str
		name of the dataset
	model_fn : list
		list with filenames for analysis
	select : str
		select what evaluation metric to use
	dir_df : str
		location to the analysis files
    
    Returns
    -------
	str
		string in csv format that contains the resulting evaluation values
    """
	ret_df = []

	# Go over all experiments
	for curr_ana in experiments:
		curr_ana_l = curr_ana.split("_")
		files_ana = select_pred_files(model_fn,curr_ana_l)
		fold_num = curr_ana_l[-2]

		print(files_ana)

		df_train,df_test,num_train,num_rep,experiment = get_df(files_ana,dir_df=dir_df)
		experiment = experiment.replace("_preds_ALL","")

		# Get the columns required
		sel_cols = [f for f in df_train.columns if f != "IDENTIFIER"]

		l1_cols_train = get_l1_cols(df_train.columns)

		l1_cols_train = [c for c in l1_cols_train if experiment in c]
		l2_cols_train = get_l2_cols(df_train.columns)
		l2_cols_train = [c for c in l2_cols_train if c in df_test.columns]

		l1_cols_train = [c for c in l1_cols_train]
		own_cols_xgb = [c for c in l1_cols_train if "xgb" in c]
		own_cols_svm = [c for c in l1_cols_train if "SVM" in c]
		own_cols_bay = [c for c in l1_cols_train if "brr" in c]
		own_cols_lass = [c for c in l1_cols_train if "lasso" in c]
		own_cols_adab = [c for c in l1_cols_train if "adaboost" in c]
		own_cols_rf = []
		
		# Do analysis based on metric selected
		if select == "cor":
			try: 
				cor_l1,cor_l2,cor_l3,cor_l4,cor_own_xgb,cor_own_svm,cor_own_bay,cor_own_adab,cor_own_lass,est_err,act_err,validation_error_train = get_cor(l1_cols_train,l2_cols_train,own_cols_xgb,own_cols_svm,own_cols_bay,own_cols_adab,own_cols_lass,df_train,df_test,experiment,fold_num=fold_num)
			except: 
                continue
			saves_scores(cor_l1,cor_l2,cor_l3,cor_l4,cor_own_xgb,cor_own_svm,cor_own_bay,cor_own_adab,cor_own_lass,num_train)
		elif select == "sumabserr": 
			cor_l1,cor_l2,cor_l3,cor_l4 = get_sumabserr(l1_cols_train,l2_cols_train,df_train,df_test)
		elif select == "abserr": 
			cor_l1,cor_l2,cor_l3,cor_l4,cor_own = get_abserr(l1_cols_train,l2_cols_train,own_cols_xgb,own_cols_svm,own_cols_bay,own_cols_lass,own_cols_rf,own_cols_adab,df_train,df_test)
			max_sc = float(max([max(df_train["time"]),max(df_test["time"])]))
			saves_scores(cor_l1/max_sc,cor_l2/max_sc,cor_l3/max_sc,cor_l4/max_sc,cor_own/max_sc,num_train)
		elif select == "avgerr":
			outfile_seerr = open("seVSerr.txt","a")
			outfile_valid = open("validationErrEstimation.txt","a")
			cor_l1,cor_l2,cor_l3,cor_l4,cor_own_xgb,cor_own_svm,cor_own_bay,cor_own_adab,cor_own_lass,est_err,act_err,validation_error_train = get_avgerr(l1_cols_train,l2_cols_train,own_cols_xgb,own_cols_svm,own_cols_bay,own_cols_adab,own_cols_lass,df_train,df_test,experiment,fold_num=fold_num)
			for inde in range(len(est_err)):
				outfile_seerr.write("%s,%s\n" % (est_err[inde],act_err[inde]))
			max_sc = float(max([max(df_train["time"]),max(df_test["time"])]))
			outfile_valid.write("%s,%s,%s\n" % (curr_ana,validation_error_train,cor_l3))
			outfile_valid.close()

			saves_scores(cor_l1/max_sc,cor_l2/max_sc,cor_l3/max_sc,cor_l4/max_sc,cor_own_xgb/max_sc,cor_own_svm/max_sc,cor_own_bay/max_sc,cor_own_adab/max_sc,cor_own_lass/max_sc,num_train)

			cor_l1 = cor_l1/max_sc
			cor_l2 = cor_l2/max_sc
			cor_l3 = cor_l3/max_sc
			cor_l4 = cor_l4/max_sc
			cor_own_xgb = cor_own_xgb/max_sc
			cor_own_svm = cor_own_svm/max_sc
			cor_own_bay = cor_own_bay/max_sc
			cor_own_adab = cor_own_adab/max_sc
			cor_own_lass = cor_own_lass/max_sc
		else: 
            continue

		ret_df.append("%s,%s,%s,%s,%s\n" % (experiment,num_rep,"Layer 1",num_train,cor_l1))
		ret_df.append("%s,%s,%s,%s,%s\n" % (experiment,num_rep,"Layer 2",num_train,cor_l2))
		ret_df.append("%s,%s,%s,%s,%s\n" % (experiment,num_rep,"Layer 3",num_train,cor_l3))
		ret_df.append("%s,%s,%s,%s,%s\n" % (experiment,num_rep,"l4",num_train,cor_l4))
		ret_df.append("%s,%s,%s,%s,%s\n" % (experiment,num_rep,"GB",num_train,cor_own_xgb))
		ret_df.append("%s,%s,%s,%s,%s\n" % (experiment,num_rep,"SVR",num_train,cor_own_svm))
		ret_df.append("%s,%s,%s,%s,%s\n" % (experiment,num_rep,"BRR",num_train,cor_own_bay))
		ret_df.append("%s,%s,%s,%s,%s\n" % (experiment,num_rep,"LASSO",num_train,cor_own_lass))
		ret_df.append("%s,%s,%s,%s,%s\n" % (experiment,num_rep,"AB",num_train,cor_own_adab))
		
		
	return(ret_df)

def get_experiments(dir_df="./Data/Predictions/duplicates/"):
	"""
    Get the names of all experiments
    
    Parameters
    ----------
	dir_df : str
		location to the analysis files
    
    Returns
    -------
	list
		all experiment names
	list
		all file names
    """
	model_fn = [f for f in listdir(dir_df) if isfile(join(dir_df, f))]
	remove = ["preds","train","l1","l2","l3"]
	experiments = []
	for f in model_fn:
		if ".csv" not in f: continue
		if "ALL" not in f: continue
		temp_f = f.split(".")[0].split("_")
		for r in remove:
			try: temp_f.remove(r)
			except: pass
		if len(temp_f) < 1: continue
		new_f = "_".join(temp_f)
		if len(new_f) < 1: continue

		experiments.append(new_f)

	experiments = list(set(experiments))
	return(experiments,model_fn)

def main(dir_df="./data/predictions/CV_dup/",output_dir="./data/parsed/CV_dup/",select="avgerr"):
	"""
    Main function for the evaluation
    
    Parameters
    ----------
	dir_df : str
		location to the analysis files
	output_dir : str
		location to the output files for the results
	select : str
		selected evaluation metrics

    Returns
    -------

    """
	global all_preds_l1
	global all_preds_l2
	global all_preds_l3

	outfile = open(os.path.join(output_dir,"results_%s.csv" % (select)),"w")
	outfile_summ = open(os.path.join(output_dir,"results_%s_Summary.csv" % (select)),"w")

	outfile.write("experiment,number_repetitions,algo,number_train,perf\n")
	outfile_summ.write("algo,number_train,perf\n")

	experiments,model_fn = get_experiments(dir_df=dir_df)
	print(experiments,model_fn)
	print("----")

	ret_df = get_preds_layers(experiments,model_fn,select=select,dir_df=dir_df)

	for ntrain in l1_scores.keys():
		outfile_summ.write("".join(["Layer 1,%s,%s\n" % (ntrain,str(x)) for x in l1_scores[ntrain]]))
		outfile_summ.write("".join(["Layer 2,%s,%s\n" % (ntrain,str(x)) for x in l2_scores[ntrain]]))
		outfile_summ.write("".join(["Layer 3,%s,%s\n" % (ntrain,str(x)) for x in l3_scores[ntrain]]))
		outfile_summ.write("".join(["Layer 4,%s,%s\n" % (ntrain,str(x)) for x in l4_scores[ntrain]]))
		outfile_summ.write("".join(["GB,%s,%s\n" % (ntrain,str(x)) for x in own_scores[ntrain]]))
		outfile_summ.write("".join(["SVR,%s,%s\n" % (ntrain,str(x)) for x in own_scores_svm[ntrain]]))
		outfile_summ.write("".join(["BRR,%s,%s\n" % (ntrain,str(x)) for x in own_scores_bay[ntrain]]))
		outfile_summ.write("".join(["AB,%s,%s\n" % (ntrain,str(x)) for x in own_scores_adab[ntrain]]))
		outfile_summ.write("".join(["LASSO,%s,%s\n" % (ntrain,str(x)) for x in own_scores_lass[ntrain]]))

	outfile_summ.close()
	outfile.write("".join(ret_df))
	outfile.close()

	all_preds_l1 = pandas.DataFrame(all_preds_l1)
	all_preds_l2 = pandas.DataFrame(all_preds_l2)
	all_preds_l3 = pandas.DataFrame(all_preds_l3)

	all_preds_l1.columns = ["rt","pred","experiment","train_size","fold_number","GB","BRR","LASSO","AB"]
	all_preds_l2.columns = ["rt","pred","experiment","train_size","fold_number"]
	all_preds_l3.columns = ["rt","pred","experiment","train_size","fold_number"]

	all_preds_l1.to_csv(os.path.join(output_dir,"full_l1_preds_%s.csv" % (select)),index=False)
	all_preds_l2.to_csv(os.path.join(output_dir,"full_l2_preds_%s.csv" % (select)),index=False)
	all_preds_l3.to_csv(os.path.join(output_dir,"full_l3_preds_%s.csv" % (select)),index=False)

def reset_vars():
    """
    Reset global variables that hold the evaluation
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
	global all_preds_l1
	global all_preds_l2
	global all_preds_l3
	global l1_scores
	global l2_scores
	global l3_scores
	global l4_scores
	global own_scores
	global own_scores_svm
	global own_scores_bay
	global own_scores_adab
	global own_scores_lass
	
	l1_scores = {}
	l2_scores = {}
	l3_scores = {}
	l4_scores = {}

	own_scores = {}
	own_scores_svm = {}
	own_scores_bay = {}
	own_scores_adab = {}
	own_scores_lass = {}

	all_preds_l1 = []
	all_preds_l2 = []
	all_preds_l3 = []


if __name__ == "__main__":
	reset_vars()
	main(dir_df="./data/predictions/aicheler_preds_2019/",output_dir="./data/parsed/aicheler_preds_2019/",select="avgerr")
	reset_vars()
	main(dir_df="./data/predictions/aicheler_preds_2019/",output_dir="./data/parsed/aicheler_preds_2019/",select="cor")
	reset_vars()

	main(dir_df="./data/predictions/dup_cv_preds_2019/",output_dir="./data/parsed/dup_cv_preds_2019/",select="avgerr")
	reset_vars()
	main(dir_df="./data/predictions/dup_preds_2019/",output_dir="./data/parsed/dup_preds_2019/",select="avgerr")
	reset_vars()
	main(dir_df="./data/predictions/nodup_cv_preds_2019_allmods/",output_dir="./data/parsed/nodup_cv_preds_2019_allmods/",select="avgerr")
	reset_vars()
	main(dir_df="./data/predictions/nodup_preds_2019_allmods/",output_dir="./data/parsed/nodup_preds_2019_allmods/",select="avgerr")
	reset_vars()

	main(dir_df="./data/predictions/dup_cv_preds_2019/",output_dir="./data/parsed/dup_cv_preds_2019/",select="cor")
	reset_vars()
	main(dir_df="./data/predictions/dup_preds_2019/",output_dir="./data/parsed/dup_preds_2019/",select="cor")
	reset_vars()
	main(dir_df="./data/predictions/nodup_cv_preds_2019_allmods/",output_dir="./data/parsed/nodup_cv_preds_2019_allmods/",select="cor")
	reset_vars()
	main(dir_df="./data/predictions/nodup_preds_2019_allmods/",output_dir="./data/parsed/nodup_preds_2019_allmods/",select="cor")
	reset_vars()
