import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')

paragraph = """Anup Jain is doing a fantastic job by
 reading and learning Gen AI with handson coding which will be helpful
  in acheieving the target of FIRE, As explained in the excel created."""


sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)

print(sentences)