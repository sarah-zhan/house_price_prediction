#  House Prices - Advanced Regression Techniques
## Predict sales prices and practice feature engineering, RFs, and gradient boosting
This project is from [Kaggle]("https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview"). You can upload your result in the Kaggle competitions and have fun!

### Practice Skills
- Creative feature engineering
- Advanced regression techniques like random forest and gradient boosting

### Data
- You can download the data from this repo "train.csv"
- "test.csv" is the test from Kaggle that you use the information to predict your sale prices and upload yours in Kaggle. I have mine "test_results.csv" for your reference.

## EDA (main.ipynb)
- use ```pandas``` to convert the dataset to dataframe to have a general idea of the dataset
- explore the data types ```dtypes```
- plot a graph to check our target value distribution ```seaborn```, ``` histplot```: it is screwed. It should use `log`
- cleaning data: change to column name and data fields to lower case, replace space with "_"
- translate some column value to their actual names, not just a representive number
- check numerical values, extreme value like 9999999999999 ```describe().round()```. ```T``` make it easier to check all the values without scrolling.
- remove useless columns for the project

## Prepare data for model training
- split the data into train, validation and test (60%, 20%, 20%)
- get final target value and change it to log value
- reset the index
- remove the target value column
- fill NaN with 0
- make the dataframe records to dictionary
- apply one-hot-encoding for categorical values

## Training models
- We don't know what model is the best fit. We will train different models and test the rmse. The model that returns the best rmse wins.

### Decision Tree
- use ```DecisionTreeRegressor()```
- compare ```y_pred```(with `X_train`) and ```y_train```, it shows ```rmse=0```. compare ```y_pred```(with `X_val`) and ```y_train```, it shows ```rmse=0.2256```.The training model is overfitting.
    **turning**
    - ```max_depth```(how many trees): pick a relative low rmse value; the result can be varied.
        - pick ```max_depth=10``` -> train rmse gets better already
    - `min_saples_leaf`(how big the tree is): set a range of max_depth from last step, in each depth, loop through a group of `min_saples_leaf`
        - make a dataframe of `"max_depth", "min_samples_leaf", "rmse"` and use seaborn to generate a heatmap; pick the best combination ```max_depth=10, min_samples_leaf=5```
