from flask import Flask, Response, render_template, request, redirect, url_for, jsonify
from joblib import load
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import pickle
import nltk
from csv import writer

pd.set_option('display.max_colwidth', 1000)

stop_words = pickle.load(open('stopwords', 'rb'))

nltk.download('wordnet')

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
        if(request.content_type == 'application/json'):
            content = request.get_json()

            if(type(content) == list):

                if(len(content) > 0):

                    title_present = True
                    plot_present = True

                    for item in content:
                        if 'title' not in item:
                            title_present = False
                        if 'plot' not in item:
                            plot_present = False
                    
                    if title_present and plot_present:
                        
                        valid_title = True
                        valid_plot = True

                        for item in content:
                            if (type(item['title']) is not str):
                                valid_title = False
                            if (type(item['plot']) is not str):
                                valid_title = False

                        if valid_title and valid_plot and len(item['title']) > 0 and len(item['plot']) > 0:
                            results = success(content)
                            print(results)
                            return jsonify(results)

                        elif not valid_title:
                            return jsonify({'msg': 'title is not str in at least one set of data'}), 422

                        elif not valid_title:
                            return jsonify({'msg': 'plot is not str in at least one set of data'}), 422

                        else:
                            return jsonify({'msg': 'invalid input data'}), 422

                    elif not title_present:
                        return jsonify({'msg': 'title missing from at least one set of data'}), 422

                    elif not plot_present:
                        return jsonify({'msg': 'plot missing from at least one set of data'}), 422
                        

                else:
                    return jsonify({'msg': 'invalid input'}), 422

            else:
                return jsonify({'msg': 'invalid input'}), 422
            #return results.to_json(orient ='records', lines= True)
        else:
            return jsonify({'msg': 'invalid input'}), 400

if __name__ == '__main__' :
    app.run(debug=True)