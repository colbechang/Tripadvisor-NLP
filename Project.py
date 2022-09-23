'''
Colbe Chang, Ian Beer, Karen Phung, Siraj Akmal
Project #2 GROUP 41
27 APR 2022
'''

import csv
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.common import apply_if_callable
import numpy as np
from scipy import stats
import seaborn as sns
import string
from sklearn.metrics import mean_squared_error
import random
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import nltk
nltk.download('stopwords')

HOTEL_DATA = "tripadvisor_hotel_reviews.csv"

# read bad words
with open('bad_words.txt', 'r') as file:
  bad_words = file.readlines()
  bad_words = [x.strip().lower() for x in bad_words]

# read good words
with open('good_words.txt', 'r') as file:
  good_words = file.readlines()
  good_words = [x.strip().lower() for x in good_words]

# read reviews
reviews = pd.read_csv('tripadvisor_hotel_reviews.csv')

# number of top words
num = 20

# put all reviews into one string then split into a list
reviews_string = ''
for index, row in reviews.iterrows():
    reviews_string += ' ' + row['Review']
reviews_string = reviews_string.split(' ')
reviews_string = [x.strip(',.!?:()$').lower() for x in reviews_string]

# count word frequencies and put them into dictionary
word_dict = {}
for i in reviews_string:
  if i not in word_dict.keys():
    word_dict[i] = 1
  else:
    word_dict[i] += 1
del word_dict['']
del word_dict["n't"]

def dict_to_list(dct):
  # turns dictionary into list of form [[key1, value1], [key2, value2], ...]
  lst = []
  for k, v in dct.items():
    lst.append([k,v])
  return lst

def sort_highlow(lst, onindex):
  return sorted(lst, key = lambda x: x[onindex], reverse=True)

def findtop(dct, n,  words_to_include = None):
  '''
  Function: returns the top n words from a list of words
  Parameters: dct (dictionary of word frequencies), 
    n (int number of top words to include), 
    words_to_include (a list of words to filter top words)
  Returns: list of lists of form [[word1, frequency1], [word2, frequency2], ...]
  '''
  if words_to_include == None:
    word_lst = dict_to_list(dct)
    word_lst = sort_highlow(word_lst,1)
    return word_lst[0:n]
  else:
    include_dct = dct.copy()
    for i in include_dct.copy().keys():
      if i not in words_to_include:
        del include_dct[i]
    word_lst = dict_to_list(include_dct)
    word_lst = sort_highlow(word_lst,1)
    return word_lst[0:n]

# top n words used in all reviews
top = findtop(word_dict, num)
topgood = findtop(word_dict, num, good_words)
topbad = findtop(word_dict, num, bad_words)


def transpose(lst):
  return [[lst[j][i] for j in range(len(lst))] for i in range(len(lst[0]))]

top = transpose(top)
topgood = transpose(topgood)
topbad = transpose(topbad)

print('Top 20', top)
print('Good:', topgood)
print('Bad:', topbad)

def plottop(lst, title):
  plt.bar(lst[0],lst[1])
  plt.title(title)
  plt.xticks(rotation = 90)
  plt.ylabel('Frequency')
  plt.xlabel('Words')
  plt.show()

# plot top frequency words
plottop(top,'20 Most Frequently Used Words')
plottop(topgood,"20 Most Frequently Used 'Good' Words")
plottop(topbad, "20 Most Frequently Used 'Bad' Words")

def get_length(string):
  """
  Function: get_length
  Parameters: string
  Returns: length of the string
  """
  return len(string)

def get_reviews(df):
    """
    Function: get_reviews
    Parameters: df - dataframe
    Returns: List of lists with each sublist having the review and its 
    corresponding rating
    """
    reviews = []
    # Changing the rating column to numbers
    df["Rating"] = df["Rating"].apply(pd.to_numeric)
    # Looping through each row in the dataframe and appending a list
    # of the review and its rating to a list
    for index, row in df.iterrows():
        reviews.append([row["Review"], row["Rating"]])
    return reviews


def clean_reviews(reviews):
    """
    Function: clean_reviews
    Parameters: reviews - List of lists with each sublist having the review and its 
    corresponding rating
    Returns: Same list of list of reviews and their ratings except each review is 
    turned into a list of words split by space with stop
    words removed
    """
    # Looping through the review list and splitting the review by space
    # and removing the stop words
    cleaned_reviews = []
    cleaned_reviews = [[reviews[i][0].split(" "), reviews[i][1]]
                       for i in range(len(reviews)) if reviews[i][0] not in stopwords.words('english')]

    return cleaned_reviews

def get_word_ratings(cleaned_reviews):
    """
    Function: get_word_ratings
    Parameters: cleaned_reviews - list of list of reviews and their corresponding rating;
    each review is a list of words with stopwords removed
    Returns: a dictionary of words as keys and their average rating as values
    """
    word_rating_dict = {}
    #Looping through the cleaned reviews list
    for i in range(len(cleaned_reviews)):
      # Looping through each word in each review
        for word in cleaned_reviews[i][0]:
          # Checking if the word is in the dictionary and filtering out non-words
          # If the word is not in the dictionary, add it and set the value
          # equal to the review's rating and start a count of how many times the word
          # appears
            if word not in word_rating_dict.keys():
                if word.isalpha():
                    word_rating_dict[word] = [cleaned_reviews[i][1], 1]
            # If the word is in the dictionary, add the rating of the review to the current total
            # and increase the count by 1
            elif word in word_rating_dict.keys():
                word_rating_dict[word][0] += cleaned_reviews[i][1]
                word_rating_dict[word][1] += 1

    # Looping through the word_rating_dict and getting the average rating of each
    # word by dividing the total rating by the count
    avg_word_ratings = {}
    for word in word_rating_dict.keys():
        avg_word_ratings[word] = word_rating_dict[word][0] / word_rating_dict[word][1]
    
    return avg_word_ratings


def get_review_rating(df, avg_word_ratings):
    """
    Function: get_review_rating
    Parameters: df - dataframe, avg_word_ratings - dictionary of words as keys
    and their average rating as values
    """
    ratings = []
    # Looping through the dataframe and getting each review and splitting it
    # into a list of words
    for index, row in df.iterrows():
        cur_rating_tot = 0
        count = 0
        review = row["Review"].split(" ")
    # Looping through each word in the review and adding the word's rating to a total
    # and increasing the count by 1 - the count represents how many words from that review
    # appear in the word rating dict
        for word in review:
            if word in avg_word_ratings.keys():
                cur_rating_tot += avg_word_ratings[word]
                count += 1
        # Getting the rating by dividing the total by the count and rounding it
        ratings.append(round(cur_rating_tot / count))
    return ratings
                
def get_accuracy(df):
    """
    Function: get_accuracy
    Parameters: df
    Returns: mean squared error of the prediction model
    """
 
    return mean_squared_error(df["Rating"], df["Predicted Rating"])

def translate(review):
    """
    Function: translate
    Parameters: review - a string 
    Returns: Same string with punctuation removed
    """
    review = review.translate(str.maketrans('','', string.punctuation))
    return review


def get_polarity(review):
    """
    Function: get_polarity
    Parameters: review - a string
    Returns: polarity of that review

    """
    pol = TextBlob(review).sentiment.polarity
    return pol

def tfidf_predict(df, split = False):
    """
    Function: tfidf_predict
    Parameters: df - dataframe
    Returns: List of predictions for each review in the dataframe
    """
    # Initiating the tfidf object to transform each review into a usuable vector
    # for training and testing
    # This model has a max features of 20000 and an ngram range of 1-3
    '''tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,3), analyzer='char', stop_words='english')
    X = tfidf.fit_transform(df["Review"])
    y = df["Rating"]'''
    
    # Creating the training and testing sets
    if split == False:
        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,3), analyzer='char', stop_words='english')
        X = tfidf.fit_transform(df["Review"])
        y = df["Rating"]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
    else:
        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,3), analyzer='char', stop_words='english')
        split[0] = tfidf.fit_transform(split[0])
        split[1] = tfidf.transform(split[1])
        
        X_train, X_test, y_train, y_test = split
    
    # Initializing the model for prediction
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    
    pred = []
    # Looping through the dataframe and predicting the rating of each review with
    # the model and appending each value to a list
    for index, row in df.iterrows():
        vec = tfidf.transform([row["Review"]])
        cur_pred = clf.predict(vec)
        pred.append(cur_pred)
    
    # The prediction returns an array of 1 value, so this is used to extract
    # that rating from the array for convenience 
    for i in range(len(pred)):
        pred[i] = pred[i][0]
        
    return pred

def tfidf_predict2(df, split = False):
    """
    Function: tfidf_predict2
    Parameters: df - dataframe
    Returns: List of predictions for each review in the dataframe
    """
    # Initiating the tfidf object to transform each review into a usuable vector
    # for training and testing
    # This model has 30000 max features and an ngram range of 2-4
    # Following steps are the same as the function above
    
    
    if split == False:
        tfidf = TfidfVectorizer(max_features=30000, ngram_range=(2,4), analyzer='char', stop_words='english')
        X = tfidf.fit_transform(df["Review"])
        y = df["Rating"]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
    else:
        tfidf = TfidfVectorizer(max_features=30000, ngram_range=(2,4), analyzer='char', stop_words='english')
        split[0] = tfidf.fit_transform(split[0])
        split[1] = tfidf.transform(split[1])
        
        X_train, X_test, y_train, y_test = split
        
    
    clf = LinearSVC() 
    clf.fit(X_train, y_train)
    
    pred = []
    for index, row in df.iterrows():
        vec = tfidf.transform([row["Review"]])
        cur_pred = clf.predict(vec)
        pred.append(cur_pred)
        
    for i in range(len(pred)):
        pred[i] = pred[i][0]
        
    return pred

def tfidf_predict3(df, split = False):
    """
    Function: tfidf_predict3
    Parameters: df - dataframe
    Returns: List of predictions for each review in the dataframe
    """
    # Initiating the tfidf object to transform each review into a usuable vector
    # for training and testing
    # This model has 10000 features and an ngram range of 1-5, as well as a balanced
    # class weight and a regularization parameter of 10
    # Following steps are the same as the function above
    
    
    if split == False:
        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,5), analyzer='char', stop_words='english')
        X = tfidf.fit_transform(df["Review"])
        y = df["Rating"]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
    else:
        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,5), analyzer='char', stop_words='english')
        split[0] = tfidf.fit_transform(split[0])
        split[1] = tfidf.transform(split[1])
        
        X_train, X_test, y_train, y_test = split
        
    clf = LinearSVC(C = 10, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    pred = []
    for index, row in df.iterrows():
        vec = tfidf.transform([row["Review"]])
        cur_pred = clf.predict(vec)
        pred.append(cur_pred)
        
    for i in range(len(pred)):
        pred[i] = pred[i][0]
        
    return pred

def vote(df):
    """
    Function: vote
    Parameters: df - dataframe
    Returns: List of predictions for each review which were voted on by 
    4 prediction models 
    """
    voted_pred = []
    # Looping through the dataframe and getting the amount of times each rating
    # is predicted by all 4 models and choosing the most common one
    # in the case of a tie, the first rating in the tie is picked
    for index, row in df.iterrows():
        ratings = [0,0,0,0,0]
        ratings[row["Prediction 1"] - 1] += 1
        ratings[row["Prediction 2"] - 1] += 1
        ratings[row["Prediction 3"] - 1] += 1
        ratings[row["Prediction 4"] - 1] += 1
        max_val = max(ratings)
        rating = ratings.index(max_val) + 1
        
        voted_pred.append(rating)
    return voted_pred

def count_ratings(df):
    """
    Function: count_ratings
    Parameters: df - dataframe
    Returns: Two lists of how many of each rating appear in the actual dataset
    and the predicted model
    """
    actual = [0,0,0,0,0]
    predicted = [0,0,0,0,0]
    
    for index, row in df.iterrows():
        actual[row["Rating"] - 1] += 1
        predicted[row["Voted Prediction"] - 1] += 1
    return actual, predicted

def main():

    # Reading in the dataset and applying functions to rows and adding new rows
    df = pd.read_csv(HOTEL_DATA)
    df["Rating"] = df["Rating"].apply(pd.to_numeric)
    df["Review Length"] = df.apply(lambda row: get_length(row["Review"]), axis=1)
    df["Review"] = df.apply(lambda row: translate(row["Review"]), axis=1)
    df["Polarity"] = df.apply(lambda row: get_polarity(row["Review"]), axis=1)
        
    # Plotting the regression between review length and rating
    print(df[["Review Length", "Rating"]].corr())
    sns.regplot(x = df["Review Length"], y = df["Rating"])
    plt.title("Rating compared to Review Length with Regression")
    #sns.heatmap()
        
        
    plt.show()
    # Plotting a scatterplot of rating compared to polarity
    sns.scatterplot(x = df["Rating"], y = df["Polarity"])
    plt.title("Rating compared to Polarity")
        
    # Getting predictions
    uncleaned_rev = get_reviews(df)
    cleaned_rev = clean_reviews(uncleaned_rev)
    word_ratings = get_word_ratings(cleaned_rev)
    rev_ratings = get_review_rating(df, word_ratings)
    df["Prediction 1"] = rev_ratings
    
    pred2 = tfidf_predict(df)
    df["Prediction 2"] = pred2
        
    pred3 = tfidf_predict2(df)
    df["Prediction 3"] = pred3
        
    pred4 = tfidf_predict3(df)
    df["Prediction 4"] = pred4
        
    # Getting voted predictions
    votes = vote(df)
    df["Voted Prediction"] = votes
    print(classification_report(df["Rating"], df["Voted Prediction"]))
    
    # Plotting a grouped bar graph of the frequency of ratings
    # between actual (blue) and predicted (orange)
    x = np.arange(1,6)
    width = .40
    y_actual, y_pred = count_ratings(df)
    plt.show()
    plt.title("Predicted vs Actual Number of Reviews by Rating")
    plt.bar(x-0.2, y_actual, width, label="Actual")
    plt.bar(x+0.2, y_pred, width, label="Predicted")
    plt.xlabel("Rating")
    plt.ylabel("Amount")
    plt.legend()
    
    plt.show()
    matrix = confusion_matrix(df["Rating"], df["Voted Prediction"])
    sns.heatmap(matrix)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Actual Rating vs Predicted Rating")
    print(df[["Rating", "Polarity"]].corr())
    
    # AI review generator
    nouns = pd.read_csv('UsedNouns.csv')
    adjectives = pd.read_csv('UsedAdj.csv')
    verbs = pd.read_csv('UsedVerbs.csv')
    adverbs = pd.read_csv('usedAdverbs.csv')

    pronouns = ['we','I','he','she','they']
    
    def convertToProbs(df):
        '''convert ['Count'] column to probabilities'''
        total = df['Count'].sum()
        df['Count'] = df['Count']/total
        
    convertToProbs(nouns)
    convertToProbs(adjectives)
    convertToProbs(verbs)
    convertToProbs(adverbs)
    
    #read reviews into DataFrame
    reviews = pd.read_csv('tripadvisor_hotel_reviews.csv')
    reviews = reviews.loc[0:5000]
    
    #clean reviews and put into one big list of words
    reviews_string = ''
    for index, row in reviews.iterrows():
        reviews_string += ' ' + row['Review']
    reviews_string = reviews_string.split(' ')
    reviews_string = [x.strip(' ,.!?:()$').lower() for x in reviews_string]
    
    reviews_string = [x for x in reviews_string if x != '']
    
    #dictionary matching word types with word type dataframe
    dct = {'adverb':adverbs, 'noun':nouns, 'adjective': adjectives, 'bverb': verbs[['Unnamed: 0', 'Base', 'Count']], 'ppverb': verbs[['Unnamed: 0','Past Participle', 'Count']]\
           , 'preverb': verbs[['Unnamed: 0', 'Present Participle', 'Count']], 'psverb':verbs[['Unnamed: 0','Past Simple', 'Count']]}
   
    #read in sentence structures
    with open('SentenceStructures.txt','r') as file:
        structures = []
        for line in file.readlines():
            structures.append(line.strip())
    
    def findLikely(word, nextType, offset):
        '''
        Function: selects the word to replace the filler word in a sentence from structures
        Parameters: word - word to be replaced, nextType - Dataframe of word type of replacement word, offset - how many 
        '''
        words = []
        for i in range(len(reviews_string)- 3):
            if reviews_string[i] == word:
                words.append(reviews_string[i + offset])
        df = nextType.copy().iloc[:,-2:]
        drops = []
        for i, v in df.iterrows():
            if str(v[1]) not in words:
                drops.append(i)
        
        df = df.drop(drops, axis = 0)
        convertToProbs(df)
    
        if df.empty == True:
            return np.random.choice(nextType.iloc[:,1], p=nextType['Count'])
        return np.random.choice(df.iloc[:,1], p=df['Count'])
    
    def generate():
        """
        Function: Generating a review using a datset of sentence structures, words, and findLikely function
        """
        review = ''
        for j in range(random.randint(3,7)):
            string = np.random.choice(structures)
            if string.find('/')>0:
                first = string.index('(')
                second = string.index(')')
                string = string.replace(string[first:second+1], np.random.choice([string[first+1:string.index('/')],string[string.index('/')+1:second]]))
    
            string = string.split(' ')
            for i in range(len(string)):
                string[i] = string[i].strip(',')
                replacement = ''
    
                    
                if string[i][1:] in dct.keys():
                    replacement = findLikely(string[i-int(string[i][0])], dct[(string[i][1:])], int(string[i][0]))
                    string[i] = str(replacement)
            sentence = ''
            for i in string:
                sentence+= ' ' + i
            sentence += '.'
            
            review+=sentence
        return review
        
        
    lst = []
    for j in range(50):
      lst.append(generate())
    
    ai = pd.DataFrame()
    ai['Review'] = lst
    ai['Rating'] = [0 for x in range(len(lst))]
    print(ai)

    # Reading in the dataset and applying functions to rows and adding new rows
    df = pd.read_csv('tripadvisor_hotel_reviews.csv').iloc[:2000]
    
    ai['Review'] = ai.apply(lambda row: translate(row["Review"]), axis=1)
    ai['Rating'] = ai["Rating"].apply(pd.to_numeric)
    ai['Review Length'] = ai.apply(lambda row: get_length(row["Review"]), axis=1)
    ai['Polarity'] = ai.apply(lambda row: get_polarity(row["Review"]), axis=1)
    
    df["Rating"] = df["Rating"].apply(pd.to_numeric)
    df["Review Length"] = df.apply(lambda row: get_length(row["Review"]), axis=1)
    df["Review"] = df.apply(lambda row: translate(row["Review"]), axis=1)
    df["Polarity"] = df.apply(lambda row: get_polarity(row["Review"]), axis=1)
    
    
    # Getting predictions
    uncleaned_rev = get_reviews(ai)
    cleaned_rev = clean_reviews(uncleaned_rev)
    word_ratings = get_word_ratings(cleaned_rev)
    rev_ratings = get_review_rating(ai, word_ratings)
    ai["Prediction 1"] = rev_ratings
    
    pred2 = tfidf_predict(ai, [df['Review'], ai['Review'], df['Rating'], ai['Rating']] )
    ai["Prediction 2"] = pred2
    
    
    pred3 = tfidf_predict2(ai, [df['Review'], ai['Review'], df['Rating'], ai['Rating']])
    ai["Prediction 3"] = pred3
    
    pred4 = tfidf_predict3(ai, [df['Review'], ai['Review'], df['Rating'], ai['Rating']])
    ai["Prediction 4"] = pred4
    
    # Getting voted predictions
    ai['Prediction'] = ai.apply(lambda x:  int((x['Prediction 1'] + x['Prediction 2'] + x['Prediction 3'] + x['Prediction 4'])/4+0.5), axis = 1)
    print(ai[['Review','Prediction']])
    
if __name__ == "__main__":
    main()

















