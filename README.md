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
    **tuning**
    - ```max_depth```(how many trees): pick a relative low rmse value; the result can be varied.
        - pick ```max_depth=10``` -> train rmse gets better already
    - `min_saples_leaf`(how big the tree is): set a range of max_depth from last step, in each depth, loop through a group of `min_saples_leaf`
        - make a dataframe of `"max_depth", "min_samples_leaf", "rmse"` and use seaborn to generate a heatmap; pick the best combination ```max_depth=10, min_samples_leaf=5```
        - calculate the rmse
    - use `feature_importances_` to check top 30 important features

### Random Forest
- `from sklearn.ensemble import RandomForestRegressor`
- pick a range from 10 to 200 to train the model
- turn it to dataframe and plot it (`n_estimators = 160` is the best), but we dont fix it yet
    **tuning**
    - `max_depth`: range [20, 30, 40, 50, 60, 70]
        - each depth, loop all the `n_estimator`
        - set the seed to fix the result `random_state=1`, help the model process faster [optional]`n_jobs=-1`
        - plot the result to find the best `max_depth`: 20
    - `min_samples_leaf`(how big the tree is): range [1, 5, 10, 15, 20]
        - each `min_samples_leaf`, loop all the `n_estimator`
        - `max_depth`: 20, set the seed to fix the result `random_state=1`, help the model process faster [optional]`n_jobs=-1`
        - plot the result to find the best `min_samples_leaf`: 1
- use `n_estimators=160, max_depth=20, min_samples_leaf=1` to train the model
- rmse result improve comparing to decision tree model

### XGBoost
- `import xgboost as xgb`
- train the model
    ```features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)```

- xgb_output(output) function to capture the output (number interation, train_rmse, val_rmse)
- plot the graph
- ```from IPython.utils.capture import capture_output
import sys``` to save the result using loop

    **tuning**
    - `eta`: ETA is the learning rate of the model. XGBoost uses gradient descent to calculate and update the model. In gradient descent, we are looking for the minimum weights that help the model to learn the data very well. This minimum weights for the features is updated each time the model passes through the features and learns the features during training. Tuning the learning rate helps you tell the model what speed it would use in deriving the minimum for the weights.
    `eta=0.3` is the best (faster and more accurate)
    - `max_depth`: how many trees? `max_depth=6` is the best.
    - `min_child_weight`: how big is the tree? `min_child_weight=10` is the best.
