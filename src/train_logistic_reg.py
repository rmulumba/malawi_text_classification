import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from nltk.tokenize import word_tokenize 
import torch
import pickle
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import config
import utils
import inference

model_name = 'log_reg_model.sav'

if __name__ == "__main__":
    train_folds = pd.read_csv("../input/train_folds.csv")
    train_folds["Text"] = train_folds["Text"].apply(utils.process_sentence)

    for fold_ in range(config.SPLITS):
        train_df = train_folds[train_folds.kfold != fold_].reset_index(drop=True)
        test_df = train_folds[train_folds.kfold == fold_].reset_index(drop=True)

        tfidf_vec = TfidfVectorizer(
            tokenizer=word_tokenize,
            token_pattern=None
            )

        #print(utils.words_to_tokenize())

        tfidf_vec.fit(utils.words_to_tokenize())

        X_train = tfidf_vec.transform(train_df.Text)
        X_test = tfidf_vec.transform(test_df.Text)
        y_train = train_df.Label
        y_test = test_df.Label

        model = LogisticRegression(
            C=1000, 
            max_iter=200, 
            solver='liblinear', 
            penalty='l2',
            random_state=42)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # calculate accuracy
        accuracy = accuracy_score(y_test, preds)
        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")

    pickle.dump(model, open(config.MODEL_PATH + config.LOG_REG_MODEL, 'wb'))
    inference.predict()
 

