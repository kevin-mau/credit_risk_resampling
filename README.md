# Credit Risk Resampling

## Overview of the Analysis

Credit risk poses a classification problem for machine learning.  The problem stems from the inbalance of healthy loans that vastly outnumber 
the amount of risky loans. In this analysis, we will train and evaluate models with imbalanced classes.  We will use a logistic regression model 
to resample the data by using the RandomOverSampler module from the imbalanced-learn library, then compare two versions of the dataset.
 
To do our comparisons, we will get the count of the target classes, train logistic regression classifiers, calculate the balanced accuracy scores, 
generate confusion matrices, and generate classification reports for both cases.

We will use a dataset of historical lending activity from a peer-to-peer lending services company to build these models.  The goal of building 
these models is so that we can identify the creditworthiness of borrowers.  The creditworthiness will be signified by a value of 0 in the 
“loan_status” column, meaning that the loan is healthy.  A value of 1 means that the loan has a high risk of defaulting.

To build the model, first we take the CSV dataset and put into a dataframe.  We will split the dataframe with 'y' being the “loan_status” column, 
and 'X' dataframe as the remaining columns.  Here we use the `value_counts` function to show us the amount of healthy loans in the dataset versus
the amount of risky loans in both the original model.  Finally, we split the data into training and testing datasets by using the function
train_test_split.  With the training datasets we can fit our logistic regression model.  We can predict and evaluate the model’s performance by
doing the following calculating the accuracy score of the model, generating a confusion matrix, and printing the classification report.

From the `value_counts` function we did earlier, we see that the 75036 number of good loans greatly outweight the 2500 risky loans, therefore we 
predict a new logistic regression rmdel with resampled training data.  We use the RandomOverSampler module from the imbalanced-learn library to 
resample the data and with `value_counts` function we confirm that the labels have an equal number of data points.  Once again, we use the 
LogisticRegression classifier, but this time on the resampled data and print and generate the same reports that we did from the testing dataset
in order to directly compare how our resampled test model did.  In the next section, we will examine these results.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

---

## Data:

The "lending_data.csv" file is a CSV file that is a dataset of historical lending activity from a peer-to-peer lending services company.

---

## Contributors

kevin-mau

---

## License

MIT
