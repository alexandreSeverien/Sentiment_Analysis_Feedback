from flair.data import Sentence
from flair.nn import Classifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



## load dataset 

with open ("./feedback_dataset.txt", "r") as file:
    comments = file.readlines()


## load model 

tagger = Classifier.load('sentiment')

## sentiment analysis function

def sentiment_analysis(text):
    '''Perform sentiment analysis on the given text.

    Parameters
    ----------
    text : str
        The input text to analyze.

    Returns
    -------
    tuple or None
        A tuple containing the predicted sentiment label and its confidence score.
        If no sentiment label is predicted, returns (None, None).
    '''
    sentence = Sentence(text)
    tagger.predict(sentence)
    if sentence.labels:
        return sentence.labels[0].value, sentence.labels[0].score
    else:
        return None, None

## iteration over my feedback dataset and store the results of the sentiment analysis function in a dictionary

results = {'POSITIVE': [],'NEGATIVE': [],'NEUTRAL': []}
for element in comments:
    sentiment, score = sentiment_analysis(element)
    if sentiment and score:
        results[sentiment].append((element, score))

## Getting the number of positive, neutral, negative comments and the total number of comments 
    
positive_count = len(results["POSITIVE"])
negative_count = len(results["NEGATIVE"])
neutral_count = len(results["NEUTRAL"])

total_comments = len(comments)

## stats

# Calculating general satisfaction rate
general_satisfaction = positive_count / total_comments

positive_values = results['POSITIVE'][:]
negative_values = results['NEGATIVE'][:]

def filter_floats(list_tuples):
    '''Filter a list of tuples to only include floating-point numbers.

    Parameters
    ----------
    list_tuples : list of tuples
        The input list of tuples.

    Returns
    -------
    list
        A list containing only the floating-point numbers from the input list of tuples.
    '''
    return [element for tuple in list_tuples for element in tuple if isinstance(element, float)]

# Extracting floating-point numbers from the positive and negative sentiment scores
positive_values = filter_floats(positive_values)
negative_values = filter_floats(negative_values)


# Calculating statistics for positive sentiment scores
average_positive = np.mean(positive_values)
median_positive = np.median(positive_values)
std_pos = np.std(positive_values)
minimum_positive = np.min(positive_values)
maximum_positive = np.max(positive_values)
q1_positive = np.percentile(positive_values, 25)
q3_positive = np.percentile(positive_values, 75)

# Calculating statistics for negative sentiment scores
average_negative = np.mean(negative_values)
median_negative = np.median(negative_values)
std_neg = np.std(negative_values)
minimum_negative = np.min(negative_values)
maximum_negative = np.max(negative_values)
q1_negative = np.percentile(negative_values, 25)
q3_negative = np.percentile(negative_values, 75)



## net promoter score

# Selecting promoters and detractors based on a threshold score
# Here, a threshold of 0.99 is chosen arbitrarily to classify sentiments as promoters or detractors
promoters_list = [element[0] for element in results["POSITIVE"] if element[1] >= 0.99]
detractors_list = [element[0] for element in results["NEGATIVE"] if element[1] >= 0.99]

# Counting the number of promoters and detractors
promoters = len(promoters_list)
detractors = len(detractors_list)

# Calculating the Net Promoter Score (NPS)
net_promoter_score = ((promoters/total_comments) * 100) - ((detractors/total_comments) * 100)

## print results

print("Total number of comments:", total_comments)
print("Positive comments:", positive_count)
print("Negative comments:", negative_count)
print("Neutral comments:", neutral_count)
print("Number of promoters (score >= 0.99):", promoters)
print("Number of detractors (score >= 0.99):", detractors)
print("NPS (Net Promoter Score):", net_promoter_score)
print("General satisfaction:", general_satisfaction)
print("\nStatistics for positive comments:")
print("Mean:", average_positive)
print("Median:", median_positive)
print("Standard deviation:", std_pos)
print("Minimum:", minimum_positive)
print("Maximum:", maximum_positive)
print("1st quartile:", q1_positive)
print("3rd quartile:", q3_positive)

print("\nStatistics for negative comments:")
print("Mean:", average_negative)
print("Median:", median_negative)
print("Standard deviation:", std_neg)
print("Minimum:", minimum_negative)
print("Maximum:", maximum_negative)
print("1st quartile:", q1_negative)
print("3rd quartile:", q3_negative)

## showcase results

# Bar plot for comment counts
labels = ['Positive', 'Negative']
counts = [positive_count, negative_count]
plt.figure(figsize=(8, 6))
plt.bar(labels, counts, color=['green', 'red'])
plt.title('Number of Comments')
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.show()

# Box plot for sentiment scores
plt.figure(figsize=(8, 6))
plt.boxplot([positive_values, negative_values], labels=['Positive', 'Negative'], patch_artist=True, boxprops=dict(facecolor='lightgrey'))
plt.title('Score Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Score')
plt.show()

# Summary statistics
stats_data = {
    'Positive': [average_positive, median_positive, std_pos, minimum_positive, maximum_positive, q1_positive, q3_positive],
    'Negative': [average_negative, median_negative, std_neg, minimum_negative, maximum_negative, q1_negative, q3_negative]
}

stats_labels = ['Mean', 'Median', 'Standard Deviation', 'Minimum', 'Maximum', '1st Quartile', '3rd Quartile']

# Net Promoter Score
plt.figure(figsize=(8, 6))
plt.bar(['Promoters', 'Detractors'], [promoters, detractors], color=['green', 'red'])
plt.title('Net Promoter Score (NPS)')
plt.xlabel('Category')
plt.ylabel('Number of Comments')
plt.show()
