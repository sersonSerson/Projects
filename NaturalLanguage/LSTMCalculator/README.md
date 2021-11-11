<p align="left">
   <img src="https://image.freepik.com/free-vector/realistic-calculator-isolated-white_153563-1.jpg"width="200">
</p>

<h1 style="text-align:center; color:#01872A; font-size:30px;background:#daf2e1;border-radius: 20px;">LSTM calculator.</h1>

# Goal: encoder-decoder architecture training and understanding.
# Size: Small (220 lines).

## 1. Problem definition

Create a Neural Network that will make sequence-to-sequence predictions in 
form of numbers multiplication.

## 2. Data.
### The data will be created manually following the logic:
1. Step 1: create a dataset ('3*5' = '15')
2. Step 2: divide into the symbols (['3', '*', '5'...)
3. Step 3: encode the symbols ([3, 12, 5, ..])
4. Step 4: convert into one-hot encoded format ([0, 0, 0, 1, ...])
5. Step 5: create LSTM model and fit it.
6. Step 6. Make a prediction.

## 3. Evaluation

## The evaluation metric chosen is Accuracy, a model should correctly predict the output value.

<img src="https://latex.codecogs.com/gif.latex?Accuracy%20%3D%20%5Cfrac%7BTrue%5C%20Positives%20&plus;%20True%5C%20Negatives%7D%7BTrue%5C%20Positives%20&plus;%20True%5C%20Negatives%20&plus;%20False%5C%20Positives%20&plus;%20False%5C%20Negatives%7D"/> 

## 4. Structure:

Contains <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/NaturalLanguage/LSTMCalculator/LSTMcalculator.ipynb">one notebook</A> with main chapters:
1. Generate data.
2. Data preprocessing.
3. Create and fit NN model.
4. Make a prediction and check the results.

## 6. Results.
## Achieved Accuracy of 1, as expected for such a simple task.
