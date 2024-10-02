# Step 1: Import the necessary module
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Step 2: Sample text data (a list of sentences)
sentences = [
    "I love programming in Python",
    "Python is great for machine learning",
    "I enjoy solving problems with code",
    "Machine learning is fascinating",
    "Code is like art"
]

# Step 3: Tokenize the sentences (Word2Vec expects tokenized text)
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Step 4: Train a Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Step 5: Check the vector for a word (e.g., 'python')
print("Word vector for 'python':")
print(model.wv['python'])

# Step 6: Find the most similar words to 'python'
print("\nWords most similar to 'python':")
print(model.wv.most_similar('python'))
