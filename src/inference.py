import pickle
import config
import utils
import os 
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def predict():
    submission_file_name = utils.next_output_file_name(config.OUTPUT_PATH)

    Le = LabelEncoder()

    train = utils.load_dataset(config.TRAIN_PATH)
    train["Label"] = Le.fit_transform(train["Label"])

    model = pickle.load(open(config.MODEL_PATH + config.LOG_REG_MODEL, 'rb'))

    test = utils.load_dataset(config.TEST_PATH)
    test['Text'] = test['Text'].apply(utils.process_sentence)

    tfidf_vec = TfidfVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None
        )

    tfidf_vec.fit(utils.words_to_tokenize())
    test_vec = tfidf_vec.transform(test.Text)

    y_preds = model.predict(test_vec)
    y_pred_transformed = Le.inverse_transform(y_preds)


    test['Label'] = y_pred_transformed
    test = test.drop('Text', axis=1)

    # print(test.head())
    test.to_csv(os.path.join(config.OUTPUT_PATH, submission_file_name), index=False)

    if os.path.exists (os.path.join(config.OUTPUT_PATH, submission_file_name)):
        print(f"File : {submission_file_name} created successfully.")
    else:
        print(f"Error creating file {submission_file_name}")