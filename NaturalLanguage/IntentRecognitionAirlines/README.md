<p align="left">
   <img src="https://image.freepik.com/free-photo/portrait-smiling-businesswoman-showing-her-boarding-pass_107420-95785.jpg"width="200">
</p>

<h1 style="text-align:center; color:#01872A; font-size:30px;background:#daf2e1;border-radius: 20px;">Intent recognition Airlines.</h1>

# Goal: intent recognition and natural language processing training.
### Size: Medium (659 lines).

## 1. Problem definition

Recognize user's actual intent given user's natural language query.

## 2. Data.
### ATIS Airline Travel Information System Dataset from Kaggle.
Source: https://www.kaggle.com/hassanamin/atis-airlinetravelinformationsystem/

| №    | Feature       | Description|
|------|:-------------:|------------------------------------------------------:|
|1     |**Question**   |User text with query about airlines.                   |
|2     |**Label**      |Actual user's intent.                                  |

## 3. Evaluation
## The evaluation metric chosen is Accuracy, basic metric for all classification tasks.

<img src="https://latex.codecogs.com/gif.latex?Accuracy%20%3D%20%5Cfrac%7BTrue%5C%20Positives%20&plus;%20True%5C%20Negatives%7D%7BTrue%5C%20Positives%20&plus;%20True%5C%20Negatives%20&plus;%20False%5C%20Positives%20&plus;%20False%5C%20Negatives%7D"/> 

## 4. Structure:

Contains <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/NaturalLanguage/IntentRecognitionAirlines/IntentRecognitionAirline.ipynb">one notebook</A> with main chapters:
1. Overview of data (EDA).
2. Establish the baseline.
3. Remove stopwords.
4. Replace airline names.
5. Normalize user text.
6. Neural networks.
7. Error analysis for NN model.
8. Metrics analysis.

## 6. Results.
### Achieved intent recognition accuracy of 0.9925.