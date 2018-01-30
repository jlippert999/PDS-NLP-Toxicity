# Overview
Python files from Portland Data Science NLP workshop: https://www.meetup.com/Portland-Data-Science-Group

## Objective
Classify toxic comments from non-toxic using Wikipedia reviewed comments.  The original data can be found here: 
http://dive-into.info/

## Background
Each comment was reviewed by ten reviewers, each of whom classified a comment from severely toxic (-2) to very helpful (2).   The data comes in two sets: unanimous comments and reviews and all comments and reviews.  The unanimous data contains reviews where all the reviewers voted the same (unanimously).

	Data files:
		Unanimous data:
			toxicity_annotated_comments_unanimous.tsv
			toxicity_annotations_unanimous.tsv
		All Comments data:
			toxicity_annotated_comments.tsv
			toxicity_annotations.tsv

	The toxicity comment files contain the following values:
		Rev_Id, Comment, Year, Logged_in, ns, sample, split
	The toxicity annotations files contain the following
		Rev_Id, worker_id, toxicity, toxicity_score

	Toxicity_Score:
		2 - very helpful
		1 - helpful
		0 - neutral 
		-1 - toxic
		-2 - severely toxic
		
	Python Files
		UnanimousFeatureAnalysis.py - analyze the unanimous data and features using various plots 
		UnanimousBatchTestClassifiers - batch classify the unanimous data to find the best classifier for the data
		AllBatchTestClassifiers.py - batch classify the full data set and find the best classifier
		AllTopClassifiersReliabilityCurve.py - using the best classifiers plot the reliability curve
		

## Process Overview
To gain an understanding of the data the unanimous files were used to review features and then batch classify to find the best classifier for the data (files: UnanimousFeatureAnalysis.py and UnanimousBatchTestClassifiers.py).  This process was repeated for the full data set (all comments data) - code files: AllBatchTestClassifiers.py and AllClassifierProbabilityCalibration.py.

## Data Cleaning and Text Preprocessing
- Read data into a Pandas Dataframe and Join the comments and reviews
- Clean the comments (remove html, non-letters, etc)
- Stem words

## Create Features
Use scikit-learn TfidfVectorizer to convert stemmed words into a matrix of features
- ngram features to create bigrams
- remove stop words
- used only top 5000 features (max_features = 5000)

## Batch Classify
Took a brute force approach and used the following scikit-learn classifiers to find the which ones produced the best results: Logistic Regression, Nearest Neighbor, Linear SVM, Gradient Boosting, Decision Tree, Random Forest, Neural Net, Multinomial NB, Bernoulli NB, SGDClassifier, Linear SVC

## Voted Classifier
After analyzing the various classifiers the Scikit-learn VotedClassifier was implement using the best classifiers

## Results
The results are saved to a csv with the original toxicity_score.  The following is the outputted results using the full data set (all comments data)
	
	Top Classifier Results
	
	            classifier train_score test_score  training_time
	4           Linear SVC       95.42      95.07       0.981164
	0  Logistic Regression       94.96      94.84       1.113694
	1        Random Forest       99.75      94.60     341.177873
	2       Multinomial NB       93.99      93.85       0.031224
	3        SGDClassifier       93.91      93.73       2.199659
	
	----------------------------------------------------
	Voted Classifier Classification Report
	             precision    recall  f1-score   support
	
	          0       0.95      0.99      0.97     28279
	          1       0.93      0.58      0.72      3650
	
	avg / total       0.95      0.95      0.94     31929
	
	Voted Classifier Confusion Matrix
	[[28114   165]
	 [ 1519  2131]]
	
	Voted Classifier Accuracy 0.94725797864
	

## TODO
- Convert python files to Jupyter notebooks
- Automate the batch classify (pick the top n classifier based on results to do final voted classification)
- Add other features (logged in, number of caps, total words, etc)
- Tweak parameters to see if that improves accuracy  (for vectorizer and classifiers) 
