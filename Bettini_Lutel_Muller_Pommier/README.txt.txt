NLP Exercise 2: Aspect-Based Sentiment Analysis

1. Authors & useful information

Authors: Florian BETTINI, Maxime LUTEL, Gabriel MULLER, Benjamin POMMIER

Libraries & versions (python 3.7.3):
- pandas : 0.24.2
- numpy : 1.16.2
- sklearn : 0.20.3


2. Model & Approach

2.1 Approach
Our goal was to use a model capable of performing aspect-based
sentiment analysis (ABSA). More precisely, we wanted to perform
a subtask of ABSA which is aspect-term sentiment analysis (ATSA).
To do so, we first focused our approach on a model developped by
Wei Xue and Tao Li. They describe their approach in the following
research paper: "Aspect Based Sentiment Analysis with Gated
Convolutional Networks", ACL, 2018. To summarize, the neural
network they created uses 2 inputs: context embeddings and target
embeddings. The first embedding is used to capture the overall
meaning of the sentence in order to predict the sentiment, while
the second embedding captures information from the target term.

After we implemented this model in Pytorch, we noted that the
neural network performed poorly on the train dataset (accuracy
of 76%). As a consequence we tried another approach: keeping the
idea behind the ATSA model (which is splitting context and
target embeddings), and implementing it with a logistic
regression in sklearn. This is the model we will present here.

2.2 Features' representation
Our final model uses 4 kinds of features:

	2.2.1 Overall information about the sentence
Here, we extracted the length, the number of upper-case letters
and the proportion of upper-case letters within the sentence.

	2.2.2 TfIdf vectorizer on the entire sentence
Here, we used a TfIdf vectorizer on the entire sentence. We set
a maximum number of features in the corpus of 1000, and a number
of n-grams from 1 (unigrams) to 3 (3-grams).

	2.2.3 TfIdf vectorizer on the context
Here, we wanted to use the first idea behind the ATSA model,
which is getting information from the context of the target
word. Hence, we define the context as being a part of the
sentence around the target term. To extract this context, we set
a window size of +/- 40 characters around the target word.
We then use, as before, a TfIdf vectorizer on the entire sentence.
We set a maximum number of features in the corpus of 1000, and a
number of n-grams from 1 (unigrams) to 3 (3-grams).

	2.2.4 TfIdf vectorizer on the target term (aspect)
Here, we wanted to use the second idea behind the ATSA model,
which is getting information from the target term within the
sentence. As in 2.2.2 and 2.2.3, we use a TfIdf vectorizer on the
target term. We set a maximum number of features in the corpus of
100, and a number of n-grams of 1 (unigrams).

2.3 Final Model
We used a logistic regression algorithm to predict the sentiments.
It uses a lbfgs solver, a l2 penalty and maximum number of
iterations of 1000. The penalization term C is set to 0.6 (meaning
a higher penalization, to ensure the replicability of our results
on the test dataset).
All these parameters were computed using a grid search and a 5-fold
cross-validation process.

2.4 Feature importance
The three best features used in order to predict each class are shown
below. S, C and A are tags to see if a word comes from respectively
the Sentence, the Context or the Target term (Aspect) vectorizer :

	2.4.1 Three best features for "negative" class
	1) length of the sentence
	2) S - bland
	3) S - horrible

	2.4.2 Three best features for "neutral" class
	1) S - ok
	2) C - ok
	3) C - average 

	2.4.3 Three best features for "positive" class
	1) C - great
	2) S - great
	3) S - delicious


3. Score on dev dataset
The accuracy on the dev dataset is 80.85%
The execution time of tester.py is 6.84 seconds.


4. Additional information
When training the logistic regression model, we tried to modify the
thresholds used to predict the sentiments. Indeed, for a multiclass
classification task, the logistic regression model computes the
probabilty of belonging to a given class. It then choose the class
with the highest computed probabilty. However, in our case, we have
an unbalanced dataset (70% of positive examples). Hence, we tried to
set different thresholds to modify the result. For example, if the
probabilty of belonging to the "negative" class is above 0.45, then
the result should be "negative", even if the probability of belonging
to the "positive" class is 0.48 (> 0.45).
We noted a small increase on train dataset results (from 81% to 81.5%
in accuracy), however these results weren't propagated on the dev
dataset (from 80.8% to 80.2%). We concluded that this lead tended to
overfit our data, and we dropped it. Hence, this functionality can be
found within our code, but is not used.