# AI4D Malawi News Classification Challenge
## Classifying news articles in Chichewa

This project is a response to Zindi's AI4D Malawi News Classification Challenge. More information about this challenge can be found [here](https://zindi.africa/competitions/ai4d-malawi-news-classification-challenge).
## Data source

The data used in this project is from the Zindi [AI4D Malawi News Classification Challenge](https://zindi.africa/competitions/ai4d-malawi-news-classification-challenge/data).

## Requirements

1. Pandas - `pip install pandas`
2. Numpy - `pip install numpy`
3. Scikit-learn - `pip install scikit-learn`
4. Torch - `pip install torch`
5. Transformers - `pip install transformers`
6. Sentencepiece - `pip install sentencepiece`
7. nltk - `pip install nltk`

## Approach

I used a pretrained xlnet-base-cased model train a classification algorithm.
## Execution

Run `train_logistic_reg.py` in the `src` directory.

## References:

1. [ Approaching (Almost) Any Machine Learning Problem - Abhishek Thakur](https://github.com/abhishekkrthakur/approachingalmost)
2. [ Top 6 Open Source Pretrained Models for Text Classification you should use](https://www.analyticsvidhya.com/blog/2020/03/6-pretrained-models-text-classification/)
3. [ Fine-tuning XLNet language model to get better results on text classification](https://medium.com/analytics-vidhya/fine-tuning-xlnet-language-model-to-get-better-results-on-text-classification-8dfb96eb49ab)
4. [ High accuracy text classification with Python](https://towardsdatascience.com/fine-tuning-bert-and-roberta-for-high-accuracy-text-classification-in-pytorch-c9e63cf64646)
5. [ Sentiment Analysis (Opinion Mining) with Python — NLP Tutorial](https://medium.com/towards-artificial-intelligence/sentiment-analysis-opinion-mining-with-python-nlp-tutorial-d1f173ca4e3c)