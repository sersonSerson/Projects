<p align="left">
   <img src="https://image.freepik.com/free-photo/robot-doing-peace-sign_1048-3527.jpg"width="200">
</p>

<h1 style="text-align:center; color:#01872A; font-size:30px;
background:#daf2e1;border-radius: 20px;">Intent recognition chatbot.</h1>

# Goal: intent recognition and natural language processing training.
### Size: Small (220 lines).

## 1. Problem definition

Create a chatbot that will understand intent of the user text and give back 
one of the predefined answers.

## 2. Data.
### Chatbots: Intent Recognition Dataset from Kaggle.
Source: https://www.kaggle.com/elvinagammed/chatbots-intent-recognition-dataset/

Data is in form of .json with key fields:

| №    | Feature       | Description|
|------|:-------------:|-------------------------------------------------------:|
|1     |**intent**     |Name of the user's intent.                              |
|2     |**text**       |Samples of the text that is appropriate for this intent.|
|3     |**responses**   |Appropriate responses for given intent.                |

## 3. Evaluation

### The output is evaluated by human to check its consciousness.

## 4. Structure:

Contains <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/NaturalLanguage/LSTMCalculator/LSTMcalculator.ipynb">one notebook</A> with main chapters:
1. Load data.
2. Preprocess data.
3. Create model.
4. Generate predictions.
5. Create service functions for chatbot.

The chatbot itself is located in [Chatbot.py](Chatbot.py) file.

## 6. Results.
### Generated a chatbot that gives reasonable answers.
Sample: 

![Alt text](chatbot.gif)
