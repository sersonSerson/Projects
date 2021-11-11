<p align="left">
   <img src="https://image.freepik.com/free-photo/physician-noting-down-symptoms-patient_53876-63308.jpg"width="200">
</p>

<h1 style="text-align:center; color:#01872A; font-size:30px;background:#daf2e1;border-radius: 20px;">Breast cancer.</h1>

# Goal: practice using Scikit learn library for classification tasks.
## Size: Medium (645 lines).

# 1. Problem definition

## Predict whether the tumor is benign or malignant given its features.

# 2. Data
## Breast Cancer Wisconsin dataset from Kaggle. Contains:

* Data.csv - data for 570 tumors with labels: Benign(B) or Malignant(M).

Source: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/

# 3. Evaluation

## The evaluation metric chosen is Recall, as Recall shows how good we can identify positive (M(malignant)) cases and immediately start responding to such serious diagnosis.

<img src="https://latex.codecogs.com/gif.latex?%5C%20%5Chuge%7BRecall%7D%20%3D%20%5Cfrac%20%7BTrue%5C%20Positives%7D%20%7BTrue%5C%20Positives%5C%20&plus;%5C%20False%5C%20Negatives%7D"/> 

# 4. Features:

Nice and visual description of different features can be found at:

https://minds.wisconsin.edu/bitstream/handle/1793/59692/TR1131.pdf;jsessionid=37F560BEC585AC4FEFEA87C35D57B340

| №    | Feature              
|------|:-------------------: 
|1     |**Radius**            
|2     |**Texture**           
|3     |**Perimeter**         
|4     |**area**              
|5     |**smoothness**        
|6     |**compactness**       
|7     |**concavity**         
|8     |**concave points**    
|9     |**symmetry**          
|10    |**fractal dimension** 

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. 

# 5. Structure:

Contains <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/Classification/BreastCancer/BreastCancer.ipynb">one notebook</A> with main chapters:
1. EDA.
2. Feature engineering.
3. Model selection and hyperparameter tuning.
4. Results analysis.

# 6. Results.
## Achieved Recall of 0.97.