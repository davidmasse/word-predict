# Word Prediction in R and Python

For the Python version of this project, please see the following blog posts, which include all code along with some background information about concepts like Zipf's Law and perplexity:

[Predicting the Next Word](https://medium.com/@davidmasse8/predicting-the-next-word-back-off-language-modeling-8db607444ba9)

[Evaluating the Model](https://medium.com/@davidmasse8/using-perplexity-to-evaluate-a-word-prediction-model-8820cf3fd3aa)

My earlier implementation in R, described below, was more complex as it had to generate a self-contained dataframe that could be queried for a real-time web app with short wait times.

## Overview

![app1](/img/app1.png)
![app3](/img/app3.png)
![app2](/img/app2.png)

[Word-Prediction App](https://immanence.shinyapps.io/shinypredict/)

The goal of this project was to build a predictive text model. Given one to three English words in sequence, this model assigns probability-like scores to candidates for the following word in general writing (see example below). The app reactively displays and graphs (no refresh button for easier use) up to the top twenty candidates and their scores.

Though any large body of human-written text could have been used for training, the model is currently trained on a random sample of textual units (separated by line breaks) from three English text files (randomly taken from Twitter, blogs and news articles, respectively) that collectively contain about 112 million words (about 840Mb loaded in R) of natural language (along with non-word strings, stray punctuation etc.). The statistical approach employed, described below, entirely ignores grammar or any notion of meaning. Rather, it tries to approximate “English,” conceived of as the set of all understandable writing in English online. This approach has the virtue of evolving over time with text-based English usage as it can learn the latest vocabulary and frequencies from training text soon after it is generated.

Parameters that can be specified to generate the dataframe that the app uses:

![parameters](/img/parameters.png)


## Prediction Example with Computation Time

![pred_ex](/img/pred_ex.png)


## Results

![results](/img/results.png)

These tests (see discussion of sampling parameters below) used 507 4-word phrases. This is after taking out 4-word phrases ending in “xx” (see discussion of “xx” below) since the model will assign zero probability to “xx” as the completion of the initial three-word phrase. For accuracy, the last word of each 4-word test phrase was compared to the first or top-three prediction given by the model’s reaction to the initial three words of the phrase. For perplexity, the formula simply uses the probabilty score assigned by the model, based on the initial three words of the phrase, to the actual final word in each phrase.

I initially wrote a recursive function to perform the core processing (see discussion of Stupid Backoff below), but despite optimization it was still at least three times slower than the clunkier nested if-else statements that I ended up using. I opted not use SQL to look up relative frequencies - just data.table functions. My function does find top candidates for the next word in a fraction of a second (at least on a desktop computer) as shown above.

Skipgrams unfortunately seemed to reduce accuracy/increase perplexity slightly (while using more memory and computation time). I am not sure whether this has to do with the way I calculate ratios of n-gram counts to (n-1)-gram counts. I tried to correct for the added frequency of certain n-grams from skipping (see code comments), but the reduced accuracy with skipgrams persisted, leading me to set aside skipgrams while leaving the code in place to accept them later if needed. The skipgram problem may have to do with the way that I handle rare and unknown words: skipgrams create many more n-grams with one or more “xx” in them (see discussion of “xx” below).


## Exploration and Sampling

The data come with subject codes identifying topics (as metadata), but we ignore these potential predictors because subject codes would not be available for a new sentence that is being composed while the predictive text algorithm is working.

The table below summarizes basic charactaristics of the three source text files, with line lengths shown as number of characters (note the maximum 140 characters per tweet as expected per Twitter’s well-known limit). "Estimated word count" divides the total number of characters per file by 5.1 (an average word length in English) and is shown in millions.

![summary_stats](/img/summary_stats.png)

We combine lines from all three sources into one fixed training “corpus” as this will give the best chance of capturing word combinations from many different contexts. A more specialized corpus could train a more accurate model (provided that accuracy is measured using a test set that has been set aside within the corpus itself rather than from an outside source), but here we will focus on the training text at hand in exploring how to model natural language.

To further examine the data under the constraints of memory and computating power, I have set a “samplerate” parameter (currently 30 %) to control the percentage of text-file lines extracted for use in training. The maximum sample rate I have been able to use is 30% (a few hours of processing to run this document). There is also a parameter (currently 10) to choose the number of lines to take from each of the three source text files to use for testing. The test lines are only used to generate 4-word phrases used to calculate accuracy and perplexity measures of the model. The test lines are selected first to create a partition, with the remaining sampled lines used for training.


## Cleaning the Raw Data

Profanity filter: we remove any lines that contain profanity (words/phrases from a standard list) so that we do not train the model to predict a profane word or word that would form a profane phrase. “Jail” and “bait” are both fine, but the latter should never come up as the most likely word to follow the former. Entire lines must be removed since removing only the offending word/phrase could leave behind words pairs that appear consecutive but were actually separated from each other. Removing these lines reduces the number of lines in our sample by up to 20%, but we can always use a slightly higher sampling rate to compensate. Any sampling bias introduced - a tendency to predict phrase continuations as they would occur in profanity-free text - would be welcome.

Using the quanteda R package, this is followed by removal of numbers, punctuation marks, symbols, separators, Twitter punctuation, hyphens and URLs. We do not eliminate stopwords (very common words that are often useless for searching) precisely because they are frequently used and should be predicted to save time when appropriate. Stemming (reducing related words to a common root form) also seems inappropriate as predicting the next stem is not helpful. This approach also does not eliminate or expand contractions, since “don’t” and “do not” are different in tone and may be more or less used in combination with different words.


## Exploring the Cleaned Data

We then employ the n-gram framework (up to 4-grams for now) from the field of natural-language processing (NLP). Lines of text are divided (“tokenized”) into single words (1-grams), consecutive pairs (2-grams), triples (3-grams) and so on.

Below we compare the distribution of n-grams for 1-, 2-, 3- and 4-grams in the training set. Counts could be used, but frequencies are more comparable since we have differing total counts (e.g. more total 2-gram instances than 1-gram instances). N-gram frequencies were found to to be linear on a log-log plot against frequency rank (Zipf distribution).

![zipf](/img/zipf.png)


## Reducing Computer Time/Memory

The negative slope is steepest for 1-grams, which have the most truly high-frequency words (vs. 2-, 3- or 4-grams). Thus we can sacrifice the most 1-grams and still cover the vast majority of all training n-gram instances. The algorithm calculates the count below which the words account for only 10 % of all word instances. At the current sample rate (30 %), this cutoff was a count of 246 instances. 7118 1-grams were kept as the vocabulary to use out of a total number of distinct words 290157.

To increase speed with little lost accuracy, 2-, 3- and 4-grams that occur only once (the majority of them) are eliminated - though this can be changed using parameters. There were 917851, 2039777 and 1543594 2-, 3- and 4-grams kept (by virtue of having two or more instances), respectively.

The mean counts of kept n-grams are 3428, 23.9, 7.2 and 4.1 for 1-, 2-, 3- and 4-grams, respectively, while the medians were 622, 4, 3 and 2. As can be seen, most n-grams occur only the minimum number of times allowed.


## Handling Rare/Unseen/Out-Of-Vocabulary Words

I believe that the low-frequency 1-grams would normally be eliminated at this point in modeling, but I changed them all to “xx,” a dummy variable to indicate a generic rare word. The higher-order n-grams - as well as the test set of 4-grams - also have words that are “rare” (in the training set) changed to “xx.” In fact these are built from the 1-grams (separately for training and test sets). “Unseen” words entered by the user are also changed to “xx,” but “xx” is never predicted as its probability weight is zeroed at end of the algorithm.


## Process and Theory

All the n-grams are then assembled into a large R data.table object with their count ratios (e.g. “with her husband” is the completion of “with her” about 5% of the time that “with her” occurs or “in the” is the completion of “in” about 15% of the time that “in” occurs). For the Shiny app, this matrix is uploaded along with an R script, which uses the Shiny package and several functions to manipulate the matrix.

The main “top” function implements Stupid Backoff (Brants, Popat, Xu, Och and Dean, 2007), which uses the maximum likelihood estimator (MLE) for the probability of any given word given the preceding words, namely the ratio of the count of the completed phrase to the count of the initial part. (Proof involves a Markov assumption and the chain rule of conditional probability.) If none is found for a particular potential final word, a discounted score (multiplied by 0.4) is assigned to the same word as the completion of a shorter initial phrase, eliminating the first word, then the second, then the third, discounting each time. All these scores are then arranged in order, the “xx” taken out, and the scores re-normalized to add up to 1 so as to retain a key property of probabilities needed for measurement of the model’s perplexity.
