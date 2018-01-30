import pandas as pd
import os
import re
import time
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
#import seaborn as sns; sns.set()


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
    #df.comment = df.comment.apply(remove_long_words)
    
    # remove rows with blank comments
    df = df[df.comment.str.len()>0]
    
    # stem
    df.comment =df.comment.apply(stem_in_place)
    
    return df


# -----------------------------------------------------------------------------
#
# process classifiers in a batch and return results
#
# -----------------------------------------------------------------------------
def batch_classify(classifiers, X_train, Y_train, X_test, Y_test):
  
    
    no_classifiers = len(classifiers.keys())

    df_results = pd.DataFrame(data=np.zeros(shape=(no_classifiers, 4)), columns = ['classifier', 'train_score', 'test_score', 'training_time'])
    predictions = {}
    count = 0
    for key, classifier in classifiers.items():
        t_start = time.clock()
        
        # classify
        classifier.fit(X_train, Y_train)
        
        t_end = time.clock()
        t_diff = t_end - t_start
        
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        
        
        pred = classifier.predict(X_test)
        
        report = classification_report(Y_test, pred)
        accuracy = metrics.accuracy_score(Y_test, pred)
                
        print('\n----------------------------------------------------')
        print(key + ' Classification Report')        
        print(report)
        print(key + ' Confusion Matrix')
        print(confusion_matrix(Y_test, pred))
        print(key + ' Accuracy', accuracy)
                        
        
        predictions[key] = pred
        df_results.loc[count,'classifier'] = key
        df_results.loc[count,'train_score'] = "{0:.2f}".format(train_score * 100)
        df_results.loc[count,'test_score'] = "{0:.2f}".format(test_score * 100)
        df_results.loc[count,'training_time'] = t_diff
                       
        count+=1
        
    return df_results, predictions




# -----------------------------------------------------------------------------
# get voted predictions
# -----------------------------------------------------------------------------
def vote_classify(votingClassifier, X_train, Y_train, X_test, Y_test):
    
    votingClassifier.fit(X_train, Y_train)  
    pred = votingClassifier.predict(X_test)  
    
    
    
    report = classification_report(Y_test, pred)
    accuracy = metrics.accuracy_score(Y_test, pred)
    
    print('\n----------------------------------------------------')
    print('Voted Classifier Classification Report')
    print(report)
    print('Voted Classifier Confusion Matrix')
    print(confusion_matrix(Y_test, pred))
    print('Voted Classifier Accuracy', accuracy)

    return pred

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


# -----------------------------------------------------------------------------
# batch classify
# -----------------------------------------------------------------------------
# classifiers to process...
test_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(n_neighbors=3, weights='distance'),
    "Linear SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators = 100),
    "Neural Net": MLPClassifier(alpha = 1),
    "Multinomial NB": MultinomialNB(),
    "Bernoulli NB": BernoulliNB(),
    "SGDClassifier": linear_model.SGDClassifier(max_iter=90),
    "Linear SVC": LinearSVC()    
    #"Gaussian NB": GaussianNB()   # requires an array (note: use .to_array)
}

# batch classify using multiple test classifiers
results_df, predictions = batch_classify(test_classifiers, X_train, Y_train, X_test, Y_test)

print("\nBatch Classify Results\n")
print(results_df.sort_values(by='test_score', ascending=False))



# -----------------------------------------------------------------------------
# classify with top 5
#   TODO: automate from top results
# -----------------------------------------------------------------------------
best_classifiers = {
    #"Logistic Regression": test_classifiers["Logistic Regression"],
    #"Nearest Neighbors": test_classifiers["Nearest Neighbors"],
    #"Linear SVM": test_classifiers["Linear SVM"],
    "Gradient Boosting": test_classifiers["Gradient Boosting"],
    "Decision Tree": test_classifiers["Decision Tree"],
    "Random Forest": test_classifiers["Random Forest"],
    #"Neural Net": test_classifiers["Neural Net"],
    #"Multinomial NB": test_classifiers["Multinomial NB"],
    #"Bernoulli NB": test_classifiers["Bernoulli NB"],
    "SGDClassifier": test_classifiers["SGDClassifier"]
    #"Linear SVC": test_classifiers["Linear SVC"]
    #"Gaussian NB": GaussianNB()   # requires an array (note: use .to_array)
}


# batch classify using multiple test classifiers
results_best_df, predictions_best = batch_classify(best_classifiers, X_train, Y_train, X_test, Y_test)

print("\nTop Classifier Results\n")
print(results_best_df.sort_values(by='test_score', ascending=False))



# -----------------------------------------------------------------------------
# process voted classifier
# -----------------------------------------------------------------------------
votingClassifier = VotingClassifier(estimators=
    [
#        ("Logistic Regression", best_classifiers["Logistic Regression"]), 
#        ("Nearest Neighbors", best_classifiers["Nearest Neighbors"]), 
#        ("Linear SVM", best_classifiers["Linear SVM"]), 
        ("Gradient Boosting", best_classifiers["Gradient Boosting"]), 
        ("Decision Tree", best_classifiers["Decision Tree"]), 
        ("Random Forest", best_classifiers["Random Forest"]), 
#        ("Neural Net", best_classifiers["Neural Net"]), 
#        ("Multinomial NB", best_classifiers["Multinomial NB"]), 
#        ("Bernoulli NB", best_classifiers["Bernoulli NB"]), 
         ("SGDClassifier", best_classifiers["SGDClassifier"])        
#        ("Linear SVC", best_classifiers["Linear SVC"])
    ], voting='hard')

# why doesn't this work?
#votingClassifier = VotingClassifier(estimators=best_classifiers, voting='hard')

# voted classify
votedPredictions = vote_classify(votingClassifier, X_train, Y_train, X_test, Y_test)





# -----------------------------------------------------------------------------
# Copy results to a dataframe
# -----------------------------------------------------------------------------
output = pd.DataFrame( data={"id":test_set_df.rev_id, 
                             "original toxicity": test_set_df.toxicity_score, 
                             #"Logistic Regression": predictions["Logistic Regression"],
                             #"Nearest Neighbors": predictions["Nearest Neighbors"],
                             "Linear SVM": predictions["Linear SVM"],
                             "Gradient Boosting": predictions["Gradient Boosting"],
                             "Decision Tree": predictions["Decision Tree"],
                             "Random Forest": predictions["Random Forest"],
                             #"Neural Net": predictions["Neural Net"],
                             #"Multinomial NB": predictions["Multinomial NB"],
                             #"Bernoulli NB": predictions["Bernoulli NB"],
                             "SGDClassifier": predictions["SGDClassifier"],
                             #"Linear SVC": predictions["Linear SVC"],
                             "Voted":  votedPredictions,
                             "zcomment":test_set_df.comment} )

# -----------------------------------------------------------------------------
# save results to csv
# -----------------------------------------------------------------------------
if (useUnanimous):
    output.to_csv('UnanimousPredictions.csv', index=False, quoting=3, escapechar='\\')
else:
    output.to_csv('AllPredictions.csv', index=False, quoting=3, escapechar='\\')





