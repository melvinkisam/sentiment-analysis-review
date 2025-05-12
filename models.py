import pandas as pd # library for reading csv file
#import nltk # library for natural language toolkit
#nltk.download('all') # this step is to download all necessary toolkit from NLTK library (since some toolkits such asstopwords is not included in the normal package)
from nltk.corpus import stopwords # library for the list of stopwords
from nltk.stem import PorterStemmer # library for stemming
from nltk.stem.wordnet import WordNetLemmatizer # library for lemmatization
import string # library for punctuations
import re # library for regular expressions 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # library for label encoding of each word (encoder)
from sklearn.model_selection import train_test_split # library to split datasets into training data and test data
from sklearn.naive_bayes import MultinomialNB # library for mutlinomial Naive Bayes model
from sklearn.ensemble import RandomForestClassifier # library for random forests model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report # library to output results

# dictionary of contraction words
contractions_dict = { 
                    "ain't": "am not",
                    "aren't": "are not",
                    "can't": "cannot",
                    "can't've": "cannot have",
                    "'cause": "because",
                    "could've": "could have",
                    "couldn't": "could not",
                    "couldn't've": "could not have",
                    "didn't": "did not",
                    "doesn't": "does not",
                    "don't": "do not",
                    "hadn't": "had not",
                    "hadn't've": "had not have",
                    "hasn't": "has not",
                    "haven't": "have not",
                    "he'd": "he would",
                    "he'd've": "he would have",
                    "he'll": "he will",
                    "he'll've": "e will have",
                    "he's": "he is",
                    "how'd": "how did",
                    "how'd'y": "how do you",
                    "how'll": "how will",
                    "how's": "how is",
                    "I'd": "I would",
                    "I'd've": "I would have",
                    "I'll": "I will",
                    "I'll've": "I will have",
                    "I'm": "I am",
                    "I've": "I have",
                    "isn't": "is not",
                    "it'd": "it would",
                    "it'd've": "it would have",
                    "it'll": "it will",
                    "it'll've": "it will have",
                    "it's": "it is",
                    "let's": "let us",
                    "ma'am": "madam",
                    "mayn't": "may not",
                    "might've": "might have",
                    "mightn't": "might not",
                    "mightn't've": "might not have",
                    "must've": "must have",
                    "mustn't": "must not",
                    "mustn't've": "must not have",
                    "needn't": "need not",
                    "needn't've": "need not have",
                    "o'clock": "of the clock",
                    "oughtn't": "ought not",
                    "oughtn't've": "ought not have",
                    "shan't": "shall not",
                    "sha'n't": "shall not",
                    "shan't've": "shall not have",
                    "she'd": "she would",
                    "she'd've": "she would have",
                    "she'll": "she will",
                    "she'll've": "she will have",
                    "she's": "she is",
                    "should've": "should have",
                    "shouldn't": "should not",
                    "shouldn't've": "should not have",
                    "so've": "so have",
                    "so's": "so as",
                    "that'd": "that would",
                    "that'd've": "that would have",
                    "that's": "that is",
                    "there'd": "there would",
                    "there'd've": "there would have",
                    "there's": "there is",
                    "they'd": "they would",
                    "they'd've": "they would have",
                    "they'll": "they will",
                    "they'll've": "they will have",
                    "they're": "they are",
                    "they've": "they have",
                    "to've": "to have",
                    "wasn't": "was not",
                    "we'd": "we would",
                    "we'd've": "we would have",
                    "we'll": "we will",
                    "we'll've": "we will have",
                    "we're": "we are",
                    "we've": "we have",
                    "weren't": "were not",
                    "what'll": "what will",
                    "what'll've": "what will have",
                    "what're": "what are",
                    "what's": "what is",
                    "what've": "what have",
                    "when's": "when is",
                    "when've": "when have",
                    "where'd": "where did",
                    "where's": "where is",
                    "where've": "where have",
                    "who'll": "who will",
                    "who'll've": "who will have",
                    "who's": "who is",
                    "who've": "who have",
                    "why's": "why is",
                    "why've": "why have",
                    "will've": "will have",
                    "won't": "will not",
                    "won't've": "will not have",
                    "would've": "would have",
                    "wouldn't": "would not",
                    "wouldn't've": "would not have",
                    "y'all": "you all",
                    "y'all'd": "you all would",
                    "y'all'd've": "you all would have",
                    "y'all're": "you all are",
                    "y'all've": "you all have",
                    "you'd": "you had",
                    "you'd've": "you would have",
                    "you'll": "you will",
                    "you'll've": "you will have",
                    "you're": "you are",
                    "you've": "you have"
                    }

# function to replace words
def replace_words(string:str, dictionary:dict):
    for k, v in dictionary.items():
        string = string.replace(k, v)
    return string

# function to remove punctuation, either stemming or lemmatizing, and stop words if exist in a sentence (pre-processing function)
def text_cleaning(a): 
    #ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    cleaning = [char for char in a if char not in string.punctuation]
    #cleaning = [ps.stem(words) for words in cleaning]
    cleaning = [lemmatizer.lemmatize(words) for words in cleaning]
    cleaning = ''.join(cleaning)
    return [word for word in cleaning.split() if word.lower() not in stopwords.words('english')]

stopWords = set(stopwords.words('english')) # list of stop words in English
#print(stopWords)

print('--------------------------------------------------------')
# first algorithm
print('Algorithm name: Multinomial Naive Bayes\n')
print('Please wait a moment while the algorithm is running...\n')

df = pd.read_csv('dataset_elec_4000.csv') # read the csv file into dataframe

# convert contractions to proper words and remove urls
df["review"] = df["review"] \
            .apply(lambda x: re.split('http:\/\/.*', str(x))[0]) \
            .str.lower() \
            .apply(lambda x: replace_words(x, contractions_dict))

x_train, x_test, y_train, y_test = train_test_split(df['review'], df['rating'], test_size=0.2) # split data into training and testing datasets

#vectorizer = CountVectorizer(analyzer=text_cleaning) # create an object that convert the list of sentences to a matrix of token counts, and apply pre-processing function
vectorizer = TfidfVectorizer(analyzer=text_cleaning) # create an object that convert the list of sentences to a matrix of token counts, and apply pre-processing function
X_train = vectorizer.fit_transform(x_train) # fit and transform the training data 
#print(X_train)
X_test = vectorizer.transform(x_test) # transform the testing data using the same mean and variance calculated from our training data
#print(X_test)
X_train = X_train.toarray() # convert the training dataframe to an array format
#print(X_train)
X_test = X_test.toarray() # convert the testing dataframe to an array format
#print(X_test)

model = MultinomialNB().fit(X_train, y_train) # create an object of the multinomial naive bayes model and fit the training dataframe into it

prediction = model.predict(X_test) # generate prediction using the model
#print(prediction) # prediction
#print(y_test) # true value

print('confusion matrix:\n' + str(confusion_matrix(y_test, prediction)) + '\n') # output the confusion matrix to evaluate the accuracy of the classification
print('Accuracy score: ' + str(accuracy_score(y_test, prediction)*float(100)) + '%') # output the accuracy of the classification
print('Precision score: ' + str(precision_score(y_test, prediction))) # output the precision of the classification (tp / (tp + fp))
print('Recall score: ' + str(recall_score(y_test, prediction))) # output the recall of the classification (tp / (tp + fn))
print('F1 score: ' + str(f1_score(y_test, prediction)) + '\n') # output the f1 score of the classification (2 * (precision * recall) / (precision + recall))
print(classification_report(y_test, prediction)) # output of classification report

print('--------------------------------------------------------')
# second algorithm
print('Algorithm name: Random forest\n')
print('Please wait a moment while the algorithm is running...\n')

model2 = RandomForestClassifier(n_estimators = 200).fit(X_train, y_train) # create an object of the random forest model and fit the training dataframe into it

prediction2 = model2.predict(X_test) # generate prediction using the model
#print(prediction2) # prediction
#print(y_test) # true value

print('confusion matrix:\n' + str(confusion_matrix(y_test, prediction2)) + '\n') # output the confusion matrix to evaluate the accuracy of the classification
print('Accuracy score: ' + str(accuracy_score(y_test, prediction2)*float(100)) + '%') # output the accuracy of the classification
print('Precision score: ' + str(precision_score(y_test, prediction2))) # output the precision of the classification (tp / (tp + fp))
print('Recall score: ' + str(recall_score(y_test, prediction2))) # output the recall of the classification (tp / (tp + fn))
print('F1 score: ' + str(f1_score(y_test, prediction2)) + '\n') # output the f1 score of the classification (2 * (precision * recall) / (precision + recall))
print(classification_report(y_test, prediction2)) # output of classification report