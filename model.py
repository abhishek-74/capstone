import pickle
import tensorflow
import pandas as pd
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# read the model objects for predictions
with open("./models/recomm.pickle", "rb") as file:
    recomm_matrix = pickle.load(file)

with open("./models/tokenizer.pickle", "rb") as file:
    tokenizer = pickle.load(file)

sentiment_model = tensorflow.keras.models.load_model('./models/sentiment')

pd_data = pd.read_csv('./data/sample30.csv')

def combine_and_clean_text(row):
    #apply text preprocessing same as used when training the DL model for sentiment

    # 1. combine reviews title with reviews text
    text = " ".join([str(row['reviews_text']),
                     str(row['reviews_title']) if row['reviews_text'] is not None else ''
                     ])

    # 2. lowercase text and remove all alphanumeric characters
    text = re.sub("[^a-z ]", "", text.lower()).replace("  ", " ")

    return text

def tokenize_and_predict(X):
    # 1. Tokenize text
    X = tokenizer.texts_to_sequences(X)

    # 2. Pad text to fix length required by the model
    X = pad_sequences(X, truncating='post', padding='post', value=0, maxlen=575)

    # 3. score with model prediction
    scores = sentiment_model.predict(X, batch_size=250)

    return scores

def row_to_json(row):
    return {
        'id': row['id'],
        'name': row['name'],
        'brand': row['brand'],
        'manufacturer': row['manufacturer'],
        'positive_perc': row['positive_perc']
    }

def get_recommendation(userid, top=20):
    # select top 20 recommended items from recommendation engine
    rec_items = pd.DataFrame(recomm_matrix.loc[userid].sort_values(ascending=False)[0:top]).reset_index()

    # get the sentiments for these top 20 selected products
    items_text = pd.merge(pd_data, rec_items, on='id', how='inner')[['id', 'brand', 'manufacturer', 'name',
                                                                      'reviews_text', 'reviews_title']]

    # preprocess text for model prediction
    items_text['clean_text'] = items_text.apply(lambda x: combine_and_clean_text(x), axis=1)

    # tokenize and pad text and apply model prediction
    items_text['sentiment_score'] = tokenize_and_predict(items_text['clean_text'].tolist())

    # apply cutoff for binary classification
    items_text['sentiment'] = items_text['sentiment_score'].apply(lambda x: 1 if x >= 0.6 else 0)

    # group by item id to get percentage of positive reviews
    items_text['positive_perc'] = items_text.groupby('id')['sentiment'].transform(lambda x: int(100 * x.sum()/x.count()))

    # sort by positive percentage and select top 5 recommended items
    reccoms = items_text.drop_duplicates('id', keep='first').sort_values('positive_perc', ascending=False).head(5)

    # convert results to json for display on UI
    return reccoms.apply(lambda x: row_to_json(x), axis=1).to_list()




