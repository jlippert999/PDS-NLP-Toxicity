import os, re
import pandas as pd
import numpy as np

import nltk
from nltk.stem import PorterStemmer
#from nltk.classify.scikitlearn import SklearnClassifier
stem = PorterStemmer()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns; sns.set()

import warnings
warnings.filterwarnings("ignore")




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

    
    # Remove non-(alpha|whitespace|apostrophe) chars, change to lowercase    
    df.comment = df.comment.apply(lambda x: re.sub("[^a-zA-Z\s']", '', x))
    df.comment = df.comment.apply(str.lower)
    
    #df.comment = df.comment.apply(remove_domain_words)  
    df.comment = df.comment.str.replace('  ', ' ')
    df.comment = df.comment.str.strip()    
    df.comment = df.comment.apply(remove_long_words)
    
    # remove rows with blank comments
    df = df[df.comment.str.len()>0]
    
    # stem
    df.comment =df.comment.apply(stem_in_place)
    
    return df




# -----------------------------------------------------------------------------    
# plot results
#  https://jmetzen.github.io/2015-04-14/calibration.html
#  ref: http://scikit-learn.org/stable/modules/calibration.html
#
def plotResultsProbabilityCalibration(classifiers, X_train, y_train, X_test, y_test):
    # <!-- collapse=True -->
    plt.figure(figsize=(9, 9))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for name, clf in classifiers.items():
    
#    for clf, name in [(lr, 'Logistic'),
#                      (gnb, 'Naive Bayes'),
#                      (svc, 'Support Vector Classification'),
#                      (rfc, 'Random Forest')]:
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)
    
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name, ))
    
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)
    
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    
    plt.tight_layout()







# -----------------------------------------------------------------------------    
# -----------------------------------------------------------------------------
# main processing
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
useUnanimous = False

#dfClean = cleanData(useUnanimous)
#dfClean.to_pickle(os.path.join('data','allCommentClean.pickle'))
dfClean = pd.read_pickle(os.path.join('data','allCommentClean.pickle'))


# only use 50%
#dfCleanPrecentToUse = dfClean.sample(frac=.5,random_state=200)
dfCleanPrecentToUse = dfClean # 100%

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


##
## NOTE: in order to plot only binary categories can be used!!!
##
Y_train = train_set_df.toxicity
Y_test = test_set_df.toxicity



# -----------------------------------------------------------------------------
# plot probabilityCalibration 
#
classifiers = {
    "Logistic Regression": LogisticRegression(),
    #"Nearest Neighbors": KNeighborsClassifier(n_neighbors=3, weights='distance'),
    #"Linear SVM": SVC(),
    #"Gradient Boosting": GradientBoostingClassifier(),
    #"Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators = 100),
    #"Neural Net": MLPClassifier(alpha = 1),
    "Multinomial NB": MultinomialNB(),
    #"Bernoulli NB": BernoulliNB(),
    "SGDClassifier": linear_model.SGDClassifier(max_iter=90),
    "Linear SVC": LinearSVC()    
    #"Gaussian NB": GaussianNB()   # requires an array (note: use .to_array)
}
plotResultsProbabilityCalibration(classifiers, X_train, Y_train, X_test, Y_test)



