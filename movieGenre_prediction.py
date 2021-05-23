from flask import Flask, render_template, request, redirect, url_for, jsonify
from joblib import load
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import nltk
from csv import writer

pd.set_option('display.max_colwidth', 1000)

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

top_genres = ['Drama',
 'Comedy',
 'Romance Film',
 'Thriller',
 'Action',
 'World cinema',
 'Crime Fiction',
 'Horror',
 'Black-and-white',
 'Indie',
 'Action/Adventure',
 'Adventure',
 'Family Film',
 'Short Film',
 'Romantic drama',
 'Animation',
 'Musical',
 'Science Fiction',
 'Mystery',
 'Romantic comedy']

def write_to_csv(input, output):
    inoutData = []
    inoutData.append(input)
    inoutData.append(output)
    with open('userMovies.csv','a') as file:
        csv_writer = writer(file)
        csv_writer.writerow(inoutData)
        file.close()

def convert_to_text(pred):
    genres = []
    for i, item in enumerate(pred):
        if(item == 1):
            genres.append(top_genres[i])
    return genres    

def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
     # split string into tokens
    tokens = nltk.tokenize.word_tokenize(text.lower())
    # only keep strings with letters
    tokens = [t for t in tokens if t.isalpha()] 
    tokens = [t for t in tokens if t not in stop_words]
    # lemmatize and stem words
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] 
    tokens = [stemmer.stem(t) for t in tokens]
    # remove short words, they're probably not useful
    tokens = [t for t in tokens if len(t) > 2]
     # remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    cleanedText = " ".join(tokens)
    return cleanedText

pipeline = load("text_classification.joblib")

def requestResults(kw):
    cleankw = clean_text(kw)
    cleankw = pd.Series(cleankw)
    predictions = pipeline.predict(cleankw)
    answer = []
    for result in predictions:
        temp = convert_to_text(result)
        answer.append(temp)
    data = str(answer)[2:-2]
    write_to_csv(kw , data)
    return data

def success(movie_data):
    for x in movie_data:
        x['genre'] = requestResults(x['plot'])    
    return movie_data


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def get_data():
    if request.method == 'POST':
        content = request.get_json()
        results = success(content)
        print(results)
        return jsonify(results)
        #return results.to_json(orient ='records', lines= True)

if __name__ == '__main__' :
    app.run(debug=True)