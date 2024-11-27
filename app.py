from flask import Flask, render_template, request, redirect, url_for, jsonify
import speech_recognition as sr
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from googlesearch import search
import pyttsx3
import random
import nltk
import re
import requests
from bs4 import BeautifulSoup

# Download the VADER lexicon if not already present
nltk.download('vader_lexicon')

# Initialize Flask app
app = Flask(__name__)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize TTS engine
engine = pyttsx3.init()

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to tokenize and stem text
def tokenize_and_stem(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Function to search Google and retrieve information
def search_google(query):
    try:
        result = next(search(query, num=1, stop=1, pause=2, tld="com"), None)
        return result
    except StopIteration:
        return None

# Function to extract meaningful information from a webpage
def extract_information(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all paragraphs and concatenate them into a single string
        paragraphs = soup.find_all('p')
        text = ' '.join([para.text for para in paragraphs])

        # Clean up the text (remove extra spaces, newlines, etc.)
        text = re.sub(r'\s+', ' ', text)
        return text[:1000]  # Limiting to 1000 characters for summary
    except Exception as e:
        print(f"Error extracting information: {e}")
        return None

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = sia.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return 'positive'
    elif sentiment['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_name = request.form.get('name')
        interaction_type = request.form.get('interaction_type')
        return render_template('chat.html', name=user_name, interaction_type=interaction_type)
    return redirect(url_for('index'))

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': "I couldn't understand that. Could you please repeat?"})
    
    tokens = tokenize_and_stem(user_input)
    query = " ".join(tokens)
    sentiment = analyze_sentiment(user_input)
    
    response = ''
    redirect_after = False

    # Define the responses for specific inputs
    if any(word in query.lower() for word in ['hello', 'hi', 'hey']):
        response = "Hey! My name is Amunet. How can I help you today?"
    elif any(word in query.lower() for word in ['who are you', 'what are you', 'tell me about yourself']):
        response = "I'm Amunet, a chatbot here to help you out. I can provide information, answer questions, and more."
    elif any(word in query.lower() for word in ['thank you', 'thanks', 'ok']):
        response = "You're welcome! Is there anything else I can help with?"
    elif any(word in query.lower() for word in ['exit', 'bye', 'goodbye']):
        response = "Goodbye! I will redirect you in 10 seconds. Feel free to ask more questions in the meantime."
        redirect_after = True
    else:
        if sentiment == 'positive':
            response = "You sound positive! How can I assist you further today?"
        elif sentiment == 'negative':
            response = "I'm sorry you're feeling down. How can I help lift your spirits?"
        else:
            search_result = search_google(user_input)
            if search_result:
                information = extract_information(search_result)
                if information:
                    response = f"Based on your query, here is some relevant information: {information}"
                else:
                    response = "I found the page, but couldn't extract relevant information."
            else:
                response = "Hmm, I couldn't find any information on that topic. Could you please ask something else?"

    return jsonify({'response': response, 'redirect_after': redirect_after})

if __name__ == '__main__':
    app.run(debug=True)
