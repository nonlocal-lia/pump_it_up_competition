# Tanzania Water Well Classification Model

## Problem

Tanzania like many developing nations has inadequate water infrustructure. Many people throughout the country depend on local pumps constructed and funded by the government, NGOs, and various private individuals. Charitable and NGo resources seeking to help improve and maintain what infructure there is would be aided by knowledge of a number of things.

* Where are there pumps that are broken or in need of repair to prevent failure?
* What features of the pumps, areas, installers, etc. tend to be correlated with pump failure?

To figure out what pumps need to be fixed or maintained you could obviously send people into the field to manually inspect all the pumps, but this gets quite expensive. Having a decent classification model can allow the government or NGOs to make predictions about hwat pumps are most likely to have problems without having to directly inspect them and thus more efficiently aallocate resources for replacements or repairs.

# The Data

The raw training data consisted of 59400 pumps with 39 recorded features beyond the id number. A full account of the cleaning process can be found in the [Cleaning and Feature Engineering notebook](./Cleaning_and_Feature_Engineering.ipynb)

### Outline of Cleaning
The mode was imputed for the missing information on whether pumps were permitted or there we public meetings

Missing lat-long and GPS height info was imputed from the mean of the smallest region that it was still known that the pump was in.

Some data entry issues in the funder and installer data was fixed and funders and installers with less than 20 pumps were grouped as "other".

Pump age and season features were engineered from the date that the pump data was recorded and the construction year o the pump.

Clearly duplicate columns or those with unacceptable overlap were droped.

# The Models

To solve this classification problem, we first constructed a series of baseline models of different types. We then runed their hyperprameters using a gridsearch and then formed a voting model from the four best performing models. A full account of the modeling process can be found in the [Modeling notebook](./Modeling.ipynb)

### Base Logistic Model
The first model constructed was a logistic model which had an accuracy of 63.4%

![logistic confusion matrix](images/base_weighted_logistic.png)

### Base Bagging Model
The first ensemble model used was a bagging model which had an accuracy of 79.8%

![bagged tree confusion matrix](images/base_bagging.png)

### BaseRandom Forest Model
The base random forest model had an accuracy of 80.5%

![random forest confusion matrix](images/base_random_forest.png)

### Base XGBoost Model
The base XGBoost model had an accuracy of 80.3%

![xgboost confusion matrix](images/base_xgboost.png)

### Base CatBoost Model
The base CatBoost model had an accuracy of 80.5%

![catboost confusion matrix](images/base_catboost.png)

### Final Voting Model
A gridsearch was performed to find better hyperparameters for the bagging, forest, XGBoost and Catboost models, these were then placed into a voting model.

The final voting model had an accuracy of 81.9%

![voting confusion matrix](images/base_voting.png)