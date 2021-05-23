import nltk
import pickle
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

pickle.dump(stop_words, open('stopwords', 'wb'))