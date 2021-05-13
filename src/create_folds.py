import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import config
import utils

# Read the training data
train = utils.load_dataset(config.TRAIN_PATH)# train = pd.read_csv("../input/Train.csv")

Le = LabelEncoder()
train["Label"] = Le.fit_transform(train["Label"])

# initialize kfold column with -1
train['kfold'] = -1

# Shuffling the training dataset
train = train.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
labels = train.Label.values

skf = StratifiedKFold(n_splits=config.SPLITS)

# Create the new folds and populate the columns
for folds, (train_, valid_) in enumerate(skf.split(X=train, y=labels)):
    train.loc[valid_, 'kfold'] = folds

train.to_csv("../input/train_folds.csv", index=False)