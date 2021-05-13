import pandas as pd
import numpy as np
import re
import os
import config
import utils

def load_dataset(path):
    return pd.read_csv(path)

def process_sentence(words):
    words = words.lower()
    words = re.sub(r'[" "]+', " ", words)
    words = re.sub(r"[^a-zA-Z0-9]+", " ", words)
    words = words.strip()
    return words

def words_to_tokenize():
    train = utils.load_dataset(config.TRAIN_PATH)
    test = utils.load_dataset(config.TEST_PATH)

    train["Text"] = train["Text"].apply(utils.process_sentence)
    test["Text"] = test["Text"].apply(utils.process_sentence)

    train_text_data = train.loc[:, 'Text']
    test_text_data = test.loc[:, 'Text']

    text_data = pd.concat([train_text_data, test_text_data], ignore_index=True)
    text_data = pd.DataFrame(text_data)
    text_data.columns = ['Text']

    return text_data.Text

def next_output_file_name(path):
    
    if len(os.walk(path).__next__()[2]) > 0:
        next_file = len(os.walk(path).__next__()[2]) + 1
    else:
        next_file = 1
    next_file_name = "submission_" + str(next_file) + ".csv"

    return next_file_name
