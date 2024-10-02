import nltk
import sklearn

paragraph = """Anup Jain is doing a fantastic job by
 reading and learning Gen AI with handson coding. it will be helpful
  in achieving the target of FIRE, As explained in the excel created.
  Anup Jain is learning and doing the fantastic job.
  He has to do a lot of hands coding inorder to fulfil the requirement.
  He has to enjoy life as well.
  """

# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()

print(X)