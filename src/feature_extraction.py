import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download NLTK resources (only the first time)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Optional for expanded lemmatizer capabilities
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stop_words]
    # Return the tokens as a string
    return " ".join(lemmatized_tokens)