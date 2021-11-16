<p align="left">
  <img src="https://image.freepik.com/free-vector/new-message-concept-landing-page_52683-26980.jpg" width="200" title="hover text">
</p>
<h1 style="text-align:center; color:#01872A; font-size:30px;background:#daf2e1;border-radius: 20px;">SMS spam.</h1>

# Goal: Practice Pipelines, ColumnTransformer and scikit-learn tools for Natural Language Processing.
## Size: Medium (910 lines).

# 1. Problem definition
## How well can we predict whether the SMS is spam?

# 2. Data
## SMS Spam Collection Dataset from Kaggle. Contains:

1. spam.csv - data for 5,574 messages with spam/non-spam labels.
Source: https://www.kaggle.com/uciml/sms-spam-collection-dataset

# 3. Evaluation
## Accuracy score - number of correctly predicted passengers divided by total number of passengers.

<img src="https://latex.codecogs.com/gif.latex?Accuracy%20%3D%20%5Cfrac%7BTrue%5C%20Positives%20&plus;%20True%5C%20Negatives%7D%7BTrue%5C%20Positives%20&plus;%20True%5C%20Negatives%20&plus;%20False%5C%20Positives%20&plus;%20False%5C%20Negatives%7D"/> 

# 4. Features:
| №   | Feature        | Description                                           |
| --- |:--------------:| -----------------------------------------------------:|
|1    |**FullMessage** |SMS message.                                           |
|2    |**Label**       | Class of the message: spam or ham (non-spam).         |

# 5. Structure:
Contains <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/NaturalLanguage/SMS%20spam/SMS%20spam.ipynb">one notebook</A> with main chapters:
1. EDA.
2. Establish baseline.
3. Preprocess data.
4. Choose model.
5. Analyze model mistakes.


# 6. Results.
Accuracy score of over 99.3%.