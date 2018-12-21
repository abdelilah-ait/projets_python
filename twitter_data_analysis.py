import twitter
import pandas as pd
import numpy as np
import datetime
import sys
import codecs
import re
import urllib
import itertools, collections
 
import nltk  
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords 
from collections import Counter  
import string  
import copy
from itertools import product, tee, combinations, chain
from nltk.stem import PorterStemmer
from operator import itemgetter # help with dataframes
 
from scipy.spatial.distance import cosine
 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream


encodingTot = sys.stdout.encoding or 'utf-8'
columns = ['Screen_Name', 'Time_Stamp', 'Tweet']
todays_date = datetime.datetime.now().date()
 
tweetDF = pd.DataFrame(columns=columns)
 
num_tweets = 500
 
for tweet in tweepy.Cursor(api.search, q="morocco,maroc,2018", lang="en",since="2018-10-01").items(num_tweets):
     
    lenDF = len(tweetDF)
 
    tweetDF.loc[lenDF] = [tweet.user.screen_name, tweet.created_at, tweet.text]
    
tweetDF.to_csv("out.csv", sep='\t', encoding = 'utf-8')
 
#tweetDF = pd.read_csv(open('C:\Windows\system32\out.csv','rU'), sep='\t', engine='c')
tweetDF = pd.read_csv('out.csv', sep='\t', engine='c')
         
tweetDF["Tweet"].head()
tweet_list_org = tweetDF['Tweet'].tolist() # convert DF to list (tweets only) NOT a nested list
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
regex_str = [
     
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-signs
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)' # other words
]
numbers = r'(?:(?:\d+,?)+(?:\.?\d+)?)'
URL = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
html_tag = r'<[^>]+>'
hash_tag = r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"
at_sign = r'(?:@[\w_]+)'
dash_quote = r"(?:[a-z][a-z'\-_]+[a-z])"
other_word = r'(?:[\w_]+)'
other_stuff = r'(?:\S)' # anything else - NOT USED
start_pound = r"([#?])(\w+)" # Start with #
start_quest_pound = r"(?:^|\s)([#?])(\w+)" # Start with ? or with #
cont_number = r'(\w*\d\w*)' # Words containing numbers
sq_br_f = r'(?:[[\w_]+)' # removes '['
sq_br_b = r'(?:][\w_]+)' # removes ']'
 
rem_bracket = r'(' + '|'.join([sq_br_f, sq_br_b]) +')'
rem_bracketC = re.compile(rem_bracket, re.VERBOSE)
 
# Removes all words of 3 characters or less *****************************************************
 
short_words = r'\W*\b\w{1,3}\b' # Short words of 3 character or less
short_wordsC = re.compile(short_words, re.VERBOSE | re.IGNORECASE)
 
# REGEX remove all words with \ and / combinations
 
slash_back =  r'\s*(?:[\w_]*\\(?:[\w_]*\\)*[\w_]*)'
slash_fwd = r'\s*(?:[\w_]*/(?:[\w_]*/)*[\w_]*)'
slash_all = r'\s*(?:[\w_]*[/\\](?:[\w_]*[/\\])*[\w_]*)'
 
# REGEX numbers, short words and URL only to EXCLUDE +++++++++++++++++++++++++++++++++++++++++++++++++++
 
num_url_short = r'(' + '|'.join([numbers, URL, short_words + sq_br_f + sq_br_b]) +')'  # Exclude from tweets
comp_num_url_short = re.compile(num_url_short, re.VERBOSE | re.IGNORECASE)
 
# Master REGEX to INCLUDE from the original tweets ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
list_regex = r'(' + '|'.join(regex_str) + ')'
 
master_regex = re.compile(list_regex, re.VERBOSE | re.IGNORECASE) # TAKE from tweets INITIALLY
def filterPick(list, filter):
    return [ ( l, m.group(1) ) for l in list for m in (filter(l),) if m]
 
search_regex = re.compile(list_regex, re.VERBOSE | re.IGNORECASE).search
 
# Use tweetList -  that is a list from DF (using .tolist())
 
outlist_init = filterPick(tweet_list_org, search_regex) # It is a tuple: initial list from all tweets
 
char_remove = [']', '[', '(', ')', '{', '}'] # characters to be removed
words_keep = ['old', 'new', 'age', 'lot', 'bag', 'top', 'cat', 'bat', 'sap', 'jda', 'tea', 'dog', 'lie', 'law', 'lab',\
             'mob', 'map', 'car', 'fat', 'sea', 'saw', 'raw', 'rob', 'win', 'can', 'get', 'fan', 'fun', 'big',\
             'use', 'pea', 'pit','pot', 'pat', 'ear', 'eye', 'kit', 'pot', 'pen', 'bud', 'bet', 'god', 'tax', 'won', 'run',\
              'lid', 'log', 'pr', 'pd', 'cop', 'nyc', 'ny', 'la', 'toy', 'war', 'law', 'lax', 'jfk', 'fed', 'cry', 'ceo',\
              'pay', 'pet', 'fan', 'fun', 'usd', 'rio']
 
emotion_list = [':)', ';)', '(:', '(;', '}', '{','}']
word_garb = ['here', 'there', 'where', 'when', 'would', 'should', 'could','thats', 'youre', 'thanks', 'hasn',\
             'thank', 'https', 'since', 'wanna', 'gonna', 'aint', 'http', 'unto', 'onto', 'into', 'havent',\
             'dont', 'done', 'cant', 'werent', 'https', 'u', 'isnt', 'go', 'theyre', 'each', 'every', 'shes', 'youve', 'youll',\
            'weve', 'theyve']
 
# Dictionary with Replacement Pairs ******************************************************************************
repl_dict = {'googleele': 'goog', 'lyin': 'lie', 'googles': 'goog', 'aapl':'apple',\
             'msft':'microsoft', 'google': 'goog', 'googl':'goog'}
 
exclude = list(string.punctuation) + emotion_list + word_garb
 
# Convert tuple to a list, then to a string; Remove the characters; Stays as a STRING. Porter Stemmer
 
stemmer=PorterStemmer()
lmtzr = WordNetLemmatizer()
# Similarity Measure *******************************************************************************************
def cosine_sim(v1, v2):
         
    rho = round(1.0 - cosine(v1, v2), 3)
    rho = rho if(not np.isnan(rho)) else 0.0
    return rho
 
# Words Replacement ***************************************************************************************
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text
 
# Function to find element with Maximum Frequency in TDM  *******************************************************************
def nanargmax(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
 
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx
 
# Define Top K Neighbours to the WORD or TWEET ***************************************************************************
def K_neighbor(k, term, list_t):
     
    # list_t - a list of tuples
    # term - value of criteria (tweet or word)
     
    neighbor = []
    neighbor = [item for item in list_t if term in item] 
    neighbor.append(item) 
     
    neighbor.sort(key = itemgetter(0), reverse=True)
       
    print ('Top  elements for '  ) 
    print(k, term)
    print ('**********************************************')
         
    for i in xrange(k):
        print (neighbor[i])
     
    return neighbor[:k]
 
# Determine Pair of Words Counter method ******************************************************************************
def Pair_words(word_list, tweet_clean_fin, n_top):
 
    pairs = list(itertools.combinations(word_list, 2)) # order does not matter
 
    #pairs = set(map(tuple, map(sorted, _pairs)))
    pairs = set(pairs)
    c = collections.Counter()
 
    for tweet in tweet_clean_fin:
        for pair in pairs:
            if pair[0] == pair[1]: 
                pass
            elif pair[0] in tweet and pair[1] in tweet:
                #c.update({pair: 1})
                c[pair] +=1
  
    return c.most_common(n_top)
 
# BIC score function ********************************************************************************
 
from sklearn import cluster
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
 
def compute_bic(kmeans,X):
    """
    Computes the BIC metric for given clusters
 
    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn
 
    X     :  multidimension np array of data points
 
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape
 
    #compute variance for all clusters beforehand
    cl_var =  (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(m)])
    const_term = 0.5 * m * np.log(N) * (d+1)
 
    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
 
    return(BIC)
# Convert tuple to a list, then to a string; Remove the characters; Stays as a STRING. Porter Stemmer
 
# Preparing CLEAN tweets tp keep SEPARATELY from WORDS in TWEETS
 
tweet_clean_fin = [] # Cleaned Tweets - Final Version
for tweet in outlist_init:
 
    tw_clean = []
    tw_clean = [ch for ch in tweet if ch not in char_remove]
 
    tw_clean = re.sub(URL, "", str(tw_clean))
    tw_clean = re.sub(html_tag, "",str(tw_clean))
    tw_clean = re.sub(hash_tag, "",str(tw_clean))
    tw_clean = re.sub(slash_all,"", str(tw_clean))
    tw_clean = re.sub(cont_number, "",str(tw_clean))
    tw_clean = re.sub(numbers, "",str(tw_clean))
    tw_clean = re.sub(start_pound, "",str(tw_clean))
    tw_clean = re.sub(start_quest_pound, "",str(tw_clean))
    tw_clean = re.sub(at_sign, "",str(tw_clean))
    tw_clean = re.sub("'", "",str(tw_clean))
    tw_clean = re.sub('"', "",str(tw_clean))
    tw_clean = re.sub(r'(?:^|\s)[@#].*?(?=[,;:.!?]|\s|$)', r'', tw_clean) # Removes # and @ in words (lookahead)
    tw_clean = lmtzr.lemmatize(str(tw_clean))
    #tw_clean = stemmer.stem(str(tw_clean))
     
    tw_clean_lst = re.findall(r'\w+', str(tw_clean))
     
    tw_clean_lst = [tw.lower() for tw in tw_clean_lst if tw.lower() not in stopwords.words('english')]
    tw_clean_lst = [word for word in tw_clean_lst if word not in exclude]
    tw_clean_lst = str([word for word in tw_clean_lst if len(word)>3 or word.lower() in words_keep])
     
    tw_clean_lst = re.findall(r'\w+', str(tw_clean_lst))
    tw_clean_lst = [replace_all(word, repl_dict) for word in tw_clean_lst]
     
    tweet_clean_fin.append(list(tw_clean_lst))
# Delete various elements from the text (LIST OF WORDS)
 
out_list_fin = []
out_string_temp = ''.join([ch for ch in str(list(outlist_init)) if ch not in char_remove])
 
out_string_temp = re.sub(URL, "", out_string_temp)
out_string_temp = re.sub(html_tag, "", out_string_temp)
out_string_temp = re.sub(hash_tag, "", out_string_temp)
out_string_temp = re.sub(slash_all,"", str(out_string_temp))
out_string_temp = re.sub(cont_number, "", out_string_temp) 
out_string_temp = re.sub(numbers, "", out_string_temp)
out_string_temp = re.sub(start_pound, "", out_string_temp)
out_string_temp = re.sub(start_quest_pound, "", out_string_temp)
out_string_temp = re.sub(at_sign, "", out_string_temp)
out_string_temp = re.sub("'", "", out_string_temp)
out_string_temp = re.sub('"', "", out_string_temp)
out_string_temp = re.sub(r'(?:^|\s)[@#].*?(?=[,;:.!?]|\s|$)', r'', out_string_temp) # Removes # and @ in words (lookahead)
 
out_list_w = re.findall(r'\w+', out_string_temp)
 
out_string_short = str([word.lower() for word in out_list_w if len(word)>3 or word.lower() in words_keep])
 
out_list_w = re.findall(r'\w+', out_string_short)   
 
out_list_w = [lmtzr.lemmatize(word) for word in out_list_w]
#out_list_w = [stemmer.stem(word) for word in out_list_w]
out_list_w = [word.lower() for word in out_list_w if word.lower() not in stopwords.words('english')]  # Remove stopwords
out_list_w = str([word.lower() for word in out_list_w if word not in exclude])
out_string_rpl = replace_all(out_list_w, repl_dict) # replace all words from dictionary
 
# Convert "Cleaned" STRING to a LIST
 
out_list_fin = re.findall(r'\w+', out_string_rpl)
 
list_len = len(out_list_fin)
word_list = set(out_list_fin) # list of unique words from all tweets - SET
word_list_len = len(word_list)
 
print ("Set = ,Original Qty = ")
print (word_list_len,list_len)
print (word_list)
print ('********************************************************************************************************')
print (tweet_clean_fin)
print (len(tweet_clean_fin))
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(tweet_clean_fin)
print("\n --- dictionary \n",dictionary)
bow_vectors = [dictionary.doc2bow(text) for text in tweet_clean_fin]

goodLdaModel=LdaModel(corpus=bow_vectors,id2word=dictionary,iterations=50,num_topics=6)
print('\n --- goodLdaModel: all topics in result ordered by significance \n')
all_goos_topics=goodLdaModel.print_topics(-1)
print(all_goos_topics)
print("\n---goodLdaModel.print_topics(num_topics=6,num_words=12 \n")
print(goodLdaModel.print_topics(num_topics=6,num_words=16))
%%time
import warnings
import pandas as  pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
fiz=plt.figure(figsize=(30,60))
for i in range(6):
    df=pd.DataFrame(goodLdaModel.show_topic(i,16),columns=['term','prob']).set_index('term')
    plt.subplot(6,3,i+1)
    plt.title('topic'+str(i+1))
    sns.barplot(x='prob',y=df.index,data=df,label='Cities',palette='Reds_d')
    plt.xlabel('probability')
plt.show()
