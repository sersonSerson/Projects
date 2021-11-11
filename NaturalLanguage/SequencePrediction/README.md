<p align="left">
   <img src="https://image.freepik.com/free-vector/write-missing-number-worksheet-education_71599-3881.jpg"width="200">
</p>

<h1 style="text-align:center; color:#01872A; font-size:30px;background:#daf2e1;border-radius: 20px;">Sequence prediction.</h1>

# Goal: encoder-decoder architecture training and understanding.
# Size: Small.

## 1. Problem definition

Create a Neural Network that will make sequence-to-sequence predictions.

## 2. Data.
### The data will be created manually following the logic:
1. Create simple 6 number sequence for inputs (training sequence):
e.g. [29, 14, 9, 28, 34, 25]
2. Create 3 number sequences for outputs that are 3 first numbers from input
sequence in reversed order (target sequence):
e.g. [9, 14, 29]
3. Create 3 number sequences for one timestep shifted outputs (sequence for 
   'Teacher Forcing' - feed it to the decoder to fasten model fitting):
e.g. [0, 9, 14]

These sequences will be encoded in one-hot encoding manner.

## 3. Evaluation

## The evaluation metric chosen is Accuracy, a model should correctly predict the output sequence.

<img src="https://latex.codecogs.com/gif.latex?Accuracy%20%3D%20%5Cfrac%7BTrue%5C%20Positives%20&plus;%20True%5C%20Negatives%7D%7BTrue%5C%20Positives%20&plus;%20True%5C%20Negatives%20&plus;%20False%5C%20Positives%20&plus;%20False%5C%20Negatives%7D"/> 

## 4. Structure:

Contains <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/NaturalLanguage/SequencePrediction/SequencePrediction.ipynb">one notebook</A> with main chapters:
1. Create data.
2. Preprocess data.
3. Create models.
4. Train model.
5. Predict and score.

## 6. Results.
## Achieved Accuracy of 1, as expected for such a simple task.
