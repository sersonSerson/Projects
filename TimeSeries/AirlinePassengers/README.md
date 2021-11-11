<p align="left">
   <img src="https://cdn.pixabay.com/photo/2016/04/30/08/35/aircraft-1362586_960_720.jpg"width="200">
</p>

<h1 style="text-align:center; color:#01872A; font-size:30px;background:#daf2e1;border-radius: 20px;">Breast cancer.</h1>

# 1. Problem definition

## Predict whether number of airline passengers given the historical data.

# 2. Data
## International Airline Passengers Dataset from Kaggle. Contains:

* international-airline-passengers.csv - data about number of airlilne 
  passengers form January 1949 to December 1960 for each month in thousands.

Source: https://www.kaggle.com/andreazzini/international-airline-passengers/

# 3. Evaluation

## The evaluation metric chosen is RMSE, which penalizes model for highly inaccurate predictions.

<img src="https://latex.codecogs.com/gif.latex?%5Chuge%20RMSE%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bn%7D%20%28y_i%20-%20%5Chat%7By%7D_i%29%5E2%7D"/> 

# 4. Features:

| №    | Feature                             | Description|
|------|:---------------------------------- :| ------------------------------------------------------------------------:|
|1     |**Month**                            |Month and year of the sample.|
|2     |**Monthly totals in thousands**      |Thousands of passengers used airlines this month.|

# 5. Structure:
Contains two notebooks:
1. LSTM Airline Passengers  - predict the number of passengers with Neural 
   Networks (LSTM model). Model was used to predict 3 years at once.
   Was not refitted with new data each month.
   
2. SARIMAX Airline Passengers - predict the number of passengers with 
   classic SARIMAX approach. Model was used to predict one month at once,
   retraining after each month (refitting with new data).
   
These notebooks have the same first chapters: 
1. EDA
2. Test harness

# 6. Results.
## Achieve following results:
1. 30.129 RMSE error for LSTM model.
2. 17.18 RMSE error for SARIMAX model.