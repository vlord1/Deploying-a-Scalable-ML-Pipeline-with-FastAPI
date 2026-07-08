# Model Card

## Model Details
This is a machine learning classification model built using a Random Forest Classifier with default hyperparameters. It was developed by a WGU student as part of a Machine Learning DevOps course. 

## Intended Use
The intended use for this model is to use census data to predict whether or not an individual's income exceeds $50k per year. This model's purpose is educational, to show the creation of an end-to-end machine learning pipeline, including model training, inference, and deployment via an API.

## Training Data
The model was trained on Census data from 1994, from the UCI Machine Learning Repository. The dataset has continuous as well as categorical features. The categorical features (like workclass, education, marital-status, occupation, relationship, race, sex, and native-country) were processed via one-hot encoding, and the target label ("salary") was processed using a label binarizer. A random sample of 80% of the dataset was used for the training phase.

## Evaluation Data
The leftover 20% of the dataset was set aside for the evaluation dataset, to test the model's performance on unseen data. The transformations used to train the data were also applied to the data's evaluation.

## Metrics
Precision, Recall, and Fbeta (F1) Score were the 3 primary metrics used to assess classification performance. When compared to the evaluation dataset, the model saw the following performance metrics:
* Precision: 0.7376
* Recall: 0.6288
* F1 Score: 0.6789

## Ethical Considerations
Some of the dataset's demographic features could be considered sensitive, such as race, sex, and native country. Because of the possibility of a machine learning model inheriting and amplifying systemic prejudices in the training data, it's possible for predictions to favor or disfavor certain subgroups. To keep an eye this, it was necessary to calculate performance metrics across categorical slices in order to track any imbalances in model accuracy between different demographics.

## Caveats and Recommendations
This is old data (1994). It would be advisable to avoid using it to represent modern-day income brackets, given inflation and more than 30 year's worth of economic shift. For practical uses, it should be retrained on census data that's more up-to-date. Furthermore, one could use hyperparameter tuning to further improve the Recall and F1 scores.