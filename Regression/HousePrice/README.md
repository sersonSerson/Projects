<p align="left">
  <img src="https://c4.wallpaperflare.com/wallpaper/378/267/803/titanic-ship-cruise-ship-drawing-night-hd-wallpaper-preview.jpg" width="200" title="hover text">
</p>
<h1 style="text-align:center; color:#01872A; font-size:30px;background:#daf2e1;border-radius: 20px;">Titanic.</h1>

# 1. Problem definition
## How well can we predict the survivors of Titanic sinking?

# 2. Data
## Titanic dataset from Kaggle. Contains:
1. Train.csv - data for 891 of 1309 passengers with info about their survival. 
   This is the training set for our models.
2. Test.csv - data for 418 of 1309 passengers with no info about survival. 
   This data will be used as a test set. Need to submit data to Kaggle to get the
   score of the prediction.

Source: https://www.kaggle.com/c/titanic

# 3. Evaluation
## Accuracy score - number of correctly predicted passengers divided by total number of passengers.

# 4. Features:
| №   | Feature       | Description                                                              |
| --- |:-------------:| ------------------------------------------------------------------------:|
|1    |**Name**       | Name of a passenger.                                                     |
|2    |**Pclass**     | Ticket class of a passenger. \nThird is the lowest, first is the highest.|
|3    |**Age**        | Age of a passenger in years.                                             |
|4    |**Fare**       | Price of a ticket.                                                       |
|5    |**Parch**      | Number of parents and children of a passenger aboard.                    |
|6    |**SibSp**      | Number of siblings and spouses of a passenger aboard.                    |
|7    |**Cabin**      | Number of a cabin occupied by passenger.                                 |
|8    |**Ticket**     | Ticket number of a passenger.                                            |
|9    |**Embarked**   | Port of embarkment. C = Cherbourg, Q = Queenstown, S = Southampton       |
|10   |**Survived**   | Whether a passenger survived.                                            |

# 5. This was a Kaggle competition.
The main target is to improve accuracy. Some decisions can seem to be a little 
complex or useless, but it still gave score improvement. 

# 6. Structure:
Contains 3 parts:

1. <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/Classification/Titanic/01%20EDA.ipynb">EDA</A>
2. <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/Classification/Titanic/02%20Feature%20Engineering.ipynb">Feature engineering</A>
3. <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/Classification/Titanic/03%20Model%20selection%20and%20Ensembles.ipynb">Model selection</A>

# 7. Results.
Score of 81.3% allowed to finish in top 2% of the contenders.