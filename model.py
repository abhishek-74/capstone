import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


#read common objects required
tokenizer = nltk.RegexpTokenizer(r"\w+")
nltk.download('stopwords')
nltk.download('wordnet')
# nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# read the model objects for predictions
with open("./models/recomm.pickle", "rb") as file:
    recomm_matrix = pickle.load(file)

with open("./models/sentiment_tfidf_vectorizer.pickle", "rb") as file:
    vectorizer = pickle.load(file)

with open("./models/sentiment.pickle", "rb") as file:
    sentiment_model = pickle.load(file)

def combine_and_clean_text(row):
    # apply text preprocessing same as used when training the DL model for sentiment
    # 1. combine reviews title with reviews text
    text = " ".join([str(row['reviews_text']),
                     str(row['reviews_title']) if row['reviews_text'] is not None else ''
                     ])
    # 2. lowercase text
    text = text.lower()

    # 3. Tokenize text
    text = tokenizer.tokenize(text)

    # 4. Remove stop words
    text = [x for x in text if x not in stop_words]

    # 5. Stem tokens
    text = [ps.stem(x) for x in text]

    # 6. Lemmatize tokens
    text = [lemmatizer.lemmatize(x) for x in text]

    return " ".join(text)

# read source data
pd_data = pd.read_csv('./data/sample30.csv')

def vectorize_and_predict(X):
    X = vectorizer.transform(X)
    scores = sentiment_model.predict(X)
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
    try:
        rec_items = pd.DataFrame(recomm_matrix.loc[userid].sort_values(ascending=False)[0:top]).reset_index()

        # get the sentiments for these top 20 selected products
        items_text = pd.merge(pd_data, rec_items, on='id', how='inner')[['id', 'brand', 'manufacturer', 'name', 'clean_text']]

        # preprocess text, step commented out as this is taking time and heroku has limited timeout period of 30 secs.
        # we have applied this step in source data already
        # pd_data['clean_text'] = pd_data.apply(lambda x: combine_and_clean_text(x), axis=1)

        # vectorize text and apply model prediction
        items_text['sentiment'] = vectorize_and_predict(items_text['clean_text'].tolist())

        # group by item id to get percentage of positive reviews
        items_text['positive_perc'] = items_text.groupby('id')['sentiment'].transform(lambda x: int(100 * x.sum()/x.count()))

        # sort by positive percentage and select top 5 recommended items
        reccoms = items_text.drop_duplicates('id', keep='first').sort_values('positive_perc', ascending=False).head(5)

        # convert results to json for display on UI
        return {'error': '', 'result': reccoms.apply(lambda x: row_to_json(x), axis=1).to_list()}

    except Exception as e1:
        return {'error': str(e1), 'result': []}



