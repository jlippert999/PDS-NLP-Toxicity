
import pandas as pd
import os
import re
import numpy as np

import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer
stem = PorterStemmer()
import itertools

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns; sns.set()

import warnings
warnings.filterwarnings("ignore")


def calc_percent_caps(someString):
    alphaOnlyValue = re.sub("[^a-zA-Z]+", "", someString)
    length = len(alphaOnlyValue)
    if (length == 0):
        return 0
    return sum(1 for c in alphaOnlyValue if c.isupper())/length

def stem_in_place(someString):
    listOfStrings = someString.split(' ')
    return ' '.join([stem.stem(word) for word in listOfStrings])

def remove_long_words(someString):
    listOfStrings = someString.split(' ')
    return ' '.join([word for word in listOfStrings if len(word) < 30])

# -----------------------------------------------------------------------------
# clean comments column
# -----------------------------------------------------------------------------
def cleanData(useUnanimous = True):
    
    # Read and merge two files into df
    if (useUnanimous):
        comments=pd.read_csv(os.path.join('data', 'toxicity_annotated_comments_unanimous.tsv'), sep='\t')
        scores=pd.read_csv(os.path.join('data','toxicity_annotations_unanimous.tsv'), sep='\t')
        
        uniqueScores = scores[["rev_id", "toxicity_score", "toxicity"]].groupby("rev_id", as_index=False).first()
        df = pd.merge(comments, uniqueScores, on="rev_id")
    
    else:
        df=pd.read_csv(os.path.join('data', 'toxicity_annotated_comments.tsv'), sep='\t')
        scores=pd.read_csv(os.path.join('data','toxicity_annotations.tsv'), sep='\t')
     
        df["mean_score"] = pd.Series(scores.groupby("rev_id",as_index=False).mean()["toxicity_score"])
        df["median_score"] = pd.Series(scores.groupby("rev_id",as_index=False).median()["toxicity_score"])

        # create catgorical variable toxicity: if mean_score < 0, toxicity=1, otherwise 0
        df["toxicity_score"] = (df["mean_score"] < 0).astype(int)
    
    # backup comment for review
    #df['original_comment'] = df.comment 
    
    # Remove HTML elements, 'NEWLINE_TOKEN', UTC
    from bs4 import BeautifulSoup
    df.comment = df.comment.apply(lambda x: BeautifulSoup(x, 'html5lib').get_text())
    df.comment = df.comment.apply(lambda x: x.replace('NEWLINE_TOKEN', ' '))
    df.comment = df.comment.apply(lambda x: x.replace('UTC', ' '))
        
    
    # add extra features...
    df.logged_in = df.logged_in.astype(int)
    df['length'] = df.comment.apply(lambda x: len(x))
    df['num_exclamation_marks'] = df.comment.apply(lambda comment: comment.count('!'))
    df['num_words'] = df.comment.apply(lambda comment: len(comment.split()))
    df['num_unique_words'] = df.comment.apply(lambda comment: len(set(w for w in comment.split())))    
    df['percent_caps'] = df.comment.apply(calc_percent_caps)
    
    # Remove non-(alpha|whitespace|apostrophe) chars, change to lowercase    
    df.comment = df.comment.apply(lambda x: re.sub("[^a-zA-Z\s']", '', x))
    df.comment = df.comment.apply(str.lower)
    
    #df.comment = df.comment.apply(remove_domain_words)  
    df.comment = df.comment.str.replace('  ', ' ')
    df.comment = df.comment.str.strip()    
    #df.comment = df.comment.apply(remove_long_words)
    
    # remove rows with blank comments
    df = df[df.comment.str.len()>0]
    
    # stem
    df.comment =df.comment.apply(stem_in_place)
    
    return df




# -----------------------------------------------------------------------------
#  
# -----------------------------------------------------------------------------
def remove_stops(someString):
    listOfStrings = someString.split(' ')
    return ' '.join([word for word in listOfStrings if word not in stopwords.words('english')])



# -----------------------------------------------------------------------------
# Creates a Frequency DF to show top word features
# -----------------------------------------------------------------------------
def createFrequencyDF(df):
    
    #dfTemp = trainDF[trainDF['toxicity'] == 0]
    df.comment = df.comment.apply(remove_stops)
    df.comment = df.comment.apply(word_tokenize)

    
    toUse = df.comment.tolist()
    toUse = list(itertools.chain(*toUse))
    freqDist = nltk.FreqDist(toUse)
    
    df = pd.DataFrame.from_dict(freqDist, orient='index')
    df['word'] = df.index
    df = df.rename(columns={0: 'Frequency'})
    df.sort_values('Frequency', ascending=False)

    return df    

# -----------------------------------------------------------------------------
# Frequency Plot
#   https://seaborn.pydata.org/tutorial/categorical.html
# -----------------------------------------------------------------------------
def frequencyPlot(df):
    #sns.countplot(x='word', data=df.sort_values('Frequency', ascending=False).head(75));
    #plt.xticks(rotation=45);

    sns.stripplot(x="word", y="Frequency", data=df.sort_values('Frequency', ascending=False).head(75), jitter=True);
    plt.xticks(rotation=90);

#############################################################################
#
def heatMapOtherFeatures(df):
    
    otherfeatures = df[['toxicity_score','logged_in','length','percent_caps','num_exclamation_marks','num_words','num_unique_words']]
    otherfeaturesMean = otherfeatures.groupby('toxicity_score').mean()
    otherfeaturesMean.corr()
    sns.heatmap(data=otherfeaturesMean.corr(), annot=True)


    sns.heatmap(data=otherfeaturesMean.corr(), annot=True)



# -----------------------------------------------------------------------------    
# -----------------------------------------------------------------------------
# main processing
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
useUnanimous = True

dfClean = cleanData(useUnanimous)

# -----------------------------------------------------------------------------
# train/test split
# -----------------------------------------------------------------------------
train_set_df, test_set_df = train_test_split(dfClean, test_size=0.2, random_state=987)
all_words_train = train_set_df.comment
all_words_test = test_set_df.comment
    

# -----------------------------------------------------------------------------
# vectorize into features
# -----------------------------------------------------------------------------
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 3),  \
                             stop_words='english',  strip_accents='unicode',  norm='l2',          
                             max_features = 5000)


X_train = vectorizer.fit_transform(all_words_train)
X_test = vectorizer.transform(all_words_test)

Y_train = train_set_df.toxicity_score
Y_test = test_set_df.toxicity_score


# to visualize matrix
vectDF = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names())


# -----------------------------------------------------------------------------
# plot word frequencies 
# -----------------------------------------------------------------------------
dfFreq = createFrequencyDF(train_set_df[train_set_df['toxicity_score'] == -2])
frequencyPlot(dfFreq)
dfFreq = createFrequencyDF(train_set_df[train_set_df['toxicity_score'] == -1])
frequencyPlot(dfFreq)
dfFreq = createFrequencyDF(train_set_df[train_set_df['toxicity_score'] == 0])
frequencyPlot(dfFreq)
dfFreq = createFrequencyDF(train_set_df[train_set_df['toxicity_score'] == 1])
frequencyPlot(dfFreq)
dfFreq = createFrequencyDF(train_set_df[train_set_df['toxicity_score'] == 2])
frequencyPlot(dfFreq)





# -----------------------------------------------------------------------------
# show heat map for added features
# -----------------------------------------------------------------------------
heatMapOtherFeatures(train_set_df)


# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------

# distribution
sns.countplot(x="toxicity_score", data=train_set_df, palette="Greens_d");


sns.pairplot(train_set_df, hue='toxicity_score')
sns.pairplot(train_set_df, vars=["logged_in", "num_words",'num_exclamation_marks','length'], hue='toxicity_score')



sns.stripplot(x='toxicity_score', y='logged_in', data=train_set_df);
sns.stripplot(x='toxicity_score', y='num_words', data=train_set_df);
sns.stripplot(x='toxicity_score', y='num_exclamation_marks', data=train_set_df);
sns.stripplot(x='toxicity_score', y='length', data=train_set_df);

sns.swarmplot(x="toxicity_score", y="logged_in", hue="toxicity_score", data=train_set_df);

sns.violinplot(x="toxicity_score", y="length", hue="toxicity_score", data=train_set_df);



#
#import matplotlib.pyplot as plt
#from numpy.random import normal
#gaussian_numbers = normal(size=1000)
#plt.hist(gaussian_numbers)
#plt.title("Gaussian Histogram")
#plt.xlabel("Value")
#plt.ylabel("Frequency")
#plt.show()
#	
#plt.hist(gaussian_numbers, bins=20, histtype='step')
