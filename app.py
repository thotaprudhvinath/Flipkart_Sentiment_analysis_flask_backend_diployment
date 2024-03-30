from flask import Flask, render_template, request
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Set up NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Set up stopwords and lemmatizer
sw = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def clean(text):
    # Remove punctuation and numbers
    text = "".join([char for char in text if char not in string.punctuation and not char.isdigit()])
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Lemmatization and remove stopwords
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in sw]
    # Join and return
    return " ".join(filtered_tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text')
        return render_template('output.html', text=text)
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load("model/naive_bayes.pkl")
    if request.method == 'POST':
        text = request.form.get('text')
        # Preprocess the input text
        text_processed = clean(text)
        # Make prediction
        prediction = model.predict([text_processed])[0]
        return render_template('output.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
