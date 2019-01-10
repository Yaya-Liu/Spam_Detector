# Spam_Detector

Purpose:
- Design a spam detection model using Naive Bayes model based on the data from Kaggle: 
https://www.kaggle.com/uciml/sms-spam-collection-dataset/version/1

- Implement the F1 Scoring algorithm and score the models performance, and answer the following questions.


1. The process of creating the model.
 
	(1) Imported data

	(2) Data clean and data process 
		- selected the first 2 columns
		- renamed columns as “label” and “text”
		- added another column and classify ham = 0 and spam = 1)

	(3) Data pre-process and normalize 
		- lowercased all letters
		- removed all punctuations
		- removed stop words
		- Applied text stemming(converting words into their root words, For example: 
		playing, play and played are sharing same root word which is play.)
		- Split a sentence to independent words.

	(4) Converted text in each row to a vector. Assign weighted values (how frequent a term occurs in all the text)
	to each vector. (weighted value means for words that occur frequently are assigned less weight.)

	(5) Split the processed dataset to training dataset (70% samples) and test dataset (30% sample).

	(6) Fitted training dataset to Naïve Bayes MultinomialNB() model.

	(7) Got predictions on test dataset by using the trained Naïve Bayes model.

	(8) Counted the value of TP, FN, TN, FP. 

	(9) Calculated accuracy and F1 score based on TP, FN, TN, FP.


2. In general, what were the features of the model.

	(1) For building the model, I prepared 2 datasets, one for training and one for testing, they have the same structure 
	but has different number of samples. 
	
	(2) Regarding the structure, they have 2 variables
		- The label, either 1(spam) or 0(ham)
		- A vector with the weighted frequency values (how frequent a term occurs in all the text). 
		
	(3) I used MultinomialNB() to train the model and get predictions based on the trained model.

3. What is the F1 Scoring formula and in what circumstances is it better then the accuracy formula?

	(1) F1 Score = 2*TP/ (All Predicted Positive + All Observation Positive). 
	    
	    Accuracy = (TP+TN)/(TP+FP+FN+TN) = (TP+TN)/All cases.

		This is because:
		F1 Score = 2*(Recall * Precision) / (Recall + Precision)
		Precision = TP/(TP+FP) = TP/All Predicted Positive
		Recall = TP/ (TP + FN) = TP/All Observation Positive

	(2) When the number of samples in each class (It is either ham or spam in this case) are not or close to equal,
	F1 score is better than the accuracy formula. For example, in this dataset, there are 4825 ham messages, 
	and 747 spam messages. Even we do not build a model and just simply say all the messages are ham, 
	we still get about 87% accuracy, but we missed out all the spam messages.  For this imbalanced dataset, 
	the accuracy of my model is about 98%, but the F1 score is 92%.  

4. What is the F1 Score of your model?
	- F1 Score on spam detection is about 92%.  




•	TP: Observation is positive, and is predicted to be positive

•	FN: Observation is positive, but is predicted negative

•	TN: Observation is negative, and is predicted to be negative

•	FP: Observation is negative, but is predicted positive
