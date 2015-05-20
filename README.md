## Kaggle Bike Sharing Demand

This Python code explores several basic machine learning approaches to the Kaggle Competition on Bike Sharing Demand. I wrote this for an assignment for the excellent Coursera "Introduction to Data Science" online course taught by Bill Howe at UW.

#### The Competition

This is a "Knowledge" competition (i.e. just for fun and practice, no prize money). The idea is to predict how many bikes will be rented each hour of a day, based on data including weather, time, temperature, whether or not its a workday, and much more. See the [Kaggle competition page](https://www.kaggle.com/c/bike-sharing-demand) for more details.

#### What this code does

This code allows the user to specify one of 10 different machine learning algorithms available from the Python [scikit-learn](http://scikit-learn.org/stable/) library, to use in predicting bike demand. The user must also specificy which data variable(s) should be used for training, and whether to 

1. train on the full training sample, in order to submit a prediction to the Kaggle competition, **OR**
2. train and test on a subset of all available data.

The first option trains the model on the full input training set, and writes the predictions to a file **output.csv**, which can be submitted directly to Kaggle. Depending on the choice of machine learning algorithm, training on the full data set might take a few minutes.

The second option splits the data (train.csv) into a training set (5% of the data) and a testing set (the remaining 95% of data). This way the accuracy of the prediction can be known immediately, without submitting to Kaggle, and the bad predictions investigated. Performance is evaluated using the [RMSLE](https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError), which is the same metric displayed as the "score" on the competition [leaderboard](https://www.kaggle.com/c/bike-sharing-demand/leaderboard).

Finally, the user has the option to visualize the data, by plotting the "count" (total number of bikes rented) as a function of a selected data variable. In order to aid in the visualization, a small random jitter (in the horizontal direction) is added to the data points in these scatter plots, so that points spread out a bit and don't lie directly on top of one another. All points are plotted (tiny cyan crosses), and the average in bins is overplotted (large black circles).


#### How to run the code

This is a python script, which can be run by typing "python bikerides.py" on the command line.

