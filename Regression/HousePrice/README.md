<p align="left">

  <img src="http://www.clipartbest.com/cliparts/yio/LdA/yioLdA9eT.jpg" width="200" title="hover text">

</p>

<h1 style="text-align:center; color:#01872A; font-size:30px;background:#daf2e1;border-radius: 20px;">House Prices.</h1>

# Goal: practice using scikit-learn and Tensorflow libraries for regression tasks.
## Size: Large (1910 lines).

# 1. Problem definition

## Predict the price of a house given its features like: area, number of rooms, area of land.

# 2. Data
## House price dataset from Kaggle. Contains:

1. Train.csv - data for 1460 houses with info about their price. 
   This is the training set for our models.

2. Test.csv - data for 1459 houses with no info about their price. 
   This data will be used as a test set. Need to submit data to Kaggle to get the 
   score of the prediction.)

Source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/

# 3. Evaluation

## The evaluation metric is Root Mean Squared Logaritmic Error (RMSLE).

<img src="https://latex.codecogs.com/gif.latex?%5Chuge%20RMSLE%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bn%7D%20%28%5Clog%28y_i%20&plus;%201%29%20-%20log%28%5Chat%7By%7D_i&plus;%201%29%29%5E2%7D"/> 

# 4. Features:

| №    | Feature       | Description|
|------|:-------------:| ------------------------------------------------------------------------:|
|1     |**SalePrice**       |The property's sale price in dollars. This is the target variable.|
|2     |**MSSubClass**      |The building class.|
|3     |**MSZoning**        |The general zoning classification.|
|4     |**LotFrontage**     |Linear feet of street connected to property.|
|5     |**LotArea**         |Lot size in square feet.|
|6     |**Street**          |Type of road access.|
|7     |**Alley**           | Type of alley access.|
|8     |**LotShape**        |General shape of property.|
|9     |**LandContour**     |Flatness of the property.|
|10    |**Utilities**       | Type of utilities available.|  
|11    |**LotConfig**       | Lot configuration|
|12    |**LandSlope**       |Slope of property.|
|13    |**Neighborhood**    |Physical locations within Ames city limits.|
|14    |**Condition1**      |Proximity to main road or railroad |
|15    |**Condition2**      |Proximity to main road or railroad (if a second is present).|
|16    |**BldgType**        |Type of dwelling.|
|17    |**HouseStyle**      |Style of dwelling.|
|18    |**OverallQual**     |Overall material and finish quality.|
|19    |**OverallCond**     |Overall condition rating.|
|20    |**YearBuilt**       |Original construction date.|
|21    |**YearRemodAdd**    |Remodel date.|
|22    |**RoofStyle**       |Type of roof.|
|23    |**RoofMatl**        |Roof material. |
|24    |**Exterior1st**     |Exterior covering on house.|
|25    |**Exterior2nd**     |Exterior covering on house (if more than one material).|
|26    |**MasVnrType**      |Masonry veneer type.|
|27    |**MasVnrArea**      |Masonry veneer area in square feet.|
|28    |**ExterQual**       |Exterior material quality.|
|29    |**ExterCond**       |Present condition of the material on the exterior.|
|30    |**Foundation**      |Type of foundation.|
|31    |**BsmtQual**        |Height of the basement.|
|32    |**BsmtCond**        |General condition of the basement.|
|33    |**BsmtExposure**    |Walkout or garden level basement walls.|
|34    |**BsmtFinType1**    |Quality of basement finished area.|
|35    |**BsmtFinSF1**      |Type 1 finished square feet.|
|36    |**BsmtFinType2**    |Quality of second finished area (if present).|
|37    |**BsmtFinSF2**      |Type 2 finished square feet.|
|38    |**BsmtUnfSF**       |Unfinished square feet of basement area.|
|39    |**TotalBsmtSF**     |Total square feet of basement area.|
|40    |**Heating**         |Type of heating.|
|41    |**HeatingQC**       |Heating quality and condition.|
|42    |**CentralAir**      |Central air conditioning.|
|43    |**Electrical**      |Electrical system.|
|44    |**1stFlrSF**        |First Floor square feet.|
|45    |**2ndFlrSF**        |Second floor square feet.|
|46    |**LowQualFinSF**    |Low quality finished square feet (all floors).|
|47    |**GrLivArea**       |Above grade (ground) living area square feet.|
|48    |**BsmtFullBath**    |Basement full bathrooms.|
|49    |**BsmtHalfBath**    |Basement half bathrooms.|
|50    |**FullBath**        |Full bathrooms above grade.|
|51    |**HalfBath**        |Half baths above grade.|
|52    |**Bedroom**         |Number of bedrooms above basement level.|
|53    |**Kitchen**         |Number of kitchens.|
|54    |**KitchenQual**     |Kitchen quality.|
|55    |**TotRmsAbvGrd**    |Total rooms above grade (does not include bathrooms).|
|56    |**Functional**      |Home functionality rating.|
|57    |**Fireplaces**      |Number of fireplaces.|
|58    |**FireplaceQu**     |Fireplace quality.|
|59    |**GarageType**      |Garage location.|
|60    |**GarageYrBlt**     |Year garage was built.|
|61    |**GarageFinish**    |Interior finish of the garage.|
|62    |**GarageCars**      |Size of garage in car capacity.|
|63    |**GarageArea**      |Size of garage in square feet.|
|64    |**GarageQual**      |Garage quality.|
|65    |**GarageCond**      |Garage condition.|
|66    |**PavedDrive**      |Paved driveway.|
|67    |**WoodDeckSF**      |Wood deck area in square feet.|
|68    |**OpenPorchSF**     |Open porch area in square feet.|
|69    |**EnclosedPorch**   |Enclosed porch area in square feet.|
|70    |**3SsnPorch**       |Three season porch area in square feet.|
|71    |**ScreenPorch**     |Screen porch area in square feet.|
|72    |**PoolArea**        |Pool area in square feet.|
|73    |**PoolQC**          |Pool quality.|
|74    |**Fence**           |Fence quality.|
|75    |**MiscFeature**     |Miscellaneous feature not covered in other categories.|
|76    |**MiscVal**         | $Value of miscellaneous feature.|
|77    |**MoSold**          |Month Sold.|
|78    |**YrSold**          |Year Sold.|
|79    |**SaleType**        | Type of sale.|
|80    |**SaleCondition**   |Condition of sale.|


# 5. This was a Kaggle competition.
The main target is to decrease error. Some decisions can seem to be a little 
complex or useless, but it still gave score improvement. 

# 6. Structure:
1. <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/Regression/HousePrice/01%20EDA.ipynb">EDA.</A>
2. <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/Regression/HousePrice/02%20Scaled%20ANN%20predictions.ipynb">ANN prediction.</A>
3. <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/Regression/HousePrice/03%20Ridge%20regression.ipynb">Ridge regression prediction.</A>
4. <A href="https://nbviewer.org/github/sersonSerson/Projects/blob/master/Regression/HousePrice/04%20Ensemble.ipynb">Ensemble prediction.</A>

# 7. Results.
## Score of 0.11732 allowed to finish in top 3% of the contenders.