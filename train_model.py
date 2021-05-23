import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, hamming_loss, classification_report, multilabel_confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams
from wordcloud import WordCloud
import neattext as nt
import neattext.functions as nfx
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss, classification_report
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump, load


warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', 300)

drive_root = './MovieSummaries'
file_path = "{}/movie.metadata.tsv".format(drive_root)
meta = pd.read_csv(file_path, sep = '\t', header = None)
meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]
file_path_2 = "{}/plot_summaries.txt".format(drive_root)
plots = []

with open(file_path_2, 'r', encoding='utf8') as f:
    reader = csv.reader(f, dialect='excel-tab') 
    for row in tqdm(reader):
        plots.append(row)

movie_id = []
plot = []

for i in tqdm(plots):
    movie_id.append(i[0])
    plot.append(i[1])

movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})

# change datatype of 'movie_id'
meta['movie_id'] = meta['movie_id'].astype(str)

# merge meta with movies
movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')

genres = []

for i in movies['genre']:
    genres.append(list(json.loads(i).values()))
    
movies['genre'] = genres

movies = movies[~(movies['genre'].str.len() == 0)]

# get all genre tags in a list
all_genres = sum(genres,[])
len(set(all_genres))

all_genres = nltk.FreqDist(all_genres)
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 'Count': list(all_genres.values())})

descGenreCount = all_genres_df.sort_values(by=['Count'], ascending=False)[0:10]
top_genres = descGenreCount['Genre'].to_list()
top_genres

movies['genre'] = movies['genre'].apply(lambda x: [word for word in x if word in top_genres])

movies = movies[~(movies['genre'].str.len() == 0)]
movies.shape

movies = movies[movies['genre'].notna()]
movies.shape

movies = movies[movies['genre'].notna()]
movies.shape

stop_words = nltk.corpus.stopwords.words('english')

movies['clean_plot'] = movies['plot'].str.lower().str.split().apply(lambda x: ' '.join([word for word in x if word not in stop_words]))

# function for text cleaning
wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()



def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
     # split string into tokens
    tokens = nltk.tokenize.word_tokenize(text.lower())
    # only keep strings with letters
    tokens = [t for t in tokens if t.isalpha()] 
    # lemmatize and stem words
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] 
    tokens = [stemmer.stem(t) for t in tokens]
    # remove short words, they're probably not useful
    tokens = [t for t in tokens if len(t) > 2]
     # remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    cleanedText = " ".join(tokens)
    return cleanedText

movies['clean_plot'] = movies['plot'].apply(lambda x: clean_text(x))


multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies['genre'])

# transform genres
y = multilabel_binarizer.transform(movies['genre'])

# Normal text tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

# Normal text train and test set
X_train, X_test, y_train, y_test = train_test_split(movies['clean_plot'], y, test_size=0.2, random_state=9)

# create TF-IDF features for normal text
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

def test_model(model, X_train, y_train, X_test, y_test):
    clf = model.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)
    print("Overall accuracy Score: {}".format(accuracy_score(y_test, clf_pred)))
    print("Overall Hamming Loss Score: {}".format(hamming_loss(y_test, clf_pred)))
    print(classification_report(y_test, clf_pred))
    return (clf, clf_pred)

svm_clf = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_normal_clf, svm_normal_pred = test_model(svm_clf, X_train_tfidf, y_train, X_test_tfidf, y_test)

def convert_to_text(pred):
    genres = []
    for i, item in enumerate(pred):
        if(item == 1):
            genres.append(top_genres[i])
    return genres    

for pred in svm_normal_pred:
    print(convert_to_text(pred))
    
pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(max_df=0.5, max_features=20000)),
                            ('model', svm_clf)])
pipeline.fit(X_train, y_train)
dump(pipeline, filename="text_classification.joblib")