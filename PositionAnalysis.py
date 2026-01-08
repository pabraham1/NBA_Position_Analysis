# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the NBA Player Stats dataset
nbaplayerstats_df = pd.read_csv('nba_dataset(NBA).csv')

# Filter out a player's season stats, who played less than 300 minutes in that season
nbaplayerstats_df = nbaplayerstats_df[nbaplayerstats_df['MP'] * nbaplayerstats_df['G'] > 300]

# Label Encode the Positions
le = LabelEncoder()
nbaplayerstats_df['Pos_Encoded'] = le.fit_transform(nbaplayerstats_df['Pos'])

'''
# Print to ensure the program is working
print(nbaplayerstats_df[['Player', 'Pos_Encoded', 'Age', 'MP', 'G', 'PTS']])
print("Dataset Printed")
'''

print("Output test 1")
X = nbaplayerstats_df[['Age', 'G', 'GS', 'MP', 'PTS', 'AST', 'ORB', 'DRB', 'STL', 'BLK', 'TOV', 'PF',
                        '3P', '3PA', '2P', '2PA', 'FG', 'FGA', 'FT', 'FTA']]

# PG Prediction
yPG = nbaplayerstats_df[['PG']]
print(yPG)

XTrain, XTest, yTrainPG, yTestPG = train_test_split(X, yPG, test_size = .3, random_state = 42)

PGRegressionModel = LogisticRegression(max_iter=2000)
PGRegressionModel.fit(XTrain , yTrainPG)

print("\nThe accuracy of the PG model is:")
print(PGRegressionModel.score(XTest, yTestPG))

print("\nPredicted Values:")
print(PGRegressionModel.predict(XTest))
print("\nActual Values:")
print(yTestPG)


# SG Prediction
ySG = nbaplayerstats_df[['SG']]

yTrainSG, yTestSG = train_test_split(ySG, test_size = .3, random_state = 42)

SGRegressionModel = LogisticRegression(max_iter=2000)
SGRegressionModel.fit(XTrain , yTrainSG)

print("\nThe accuracy of the SG model is:")
print(SGRegressionModel.score(XTest, yTestSG))

print("\nPredicted Values:")
print(SGRegressionModel.predict(XTest))
print("\nActual Values:")
print(yTestSG)

# SF Prediction
ySF = nbaplayerstats_df[['SF']]

yTrainSF, yTestSF = train_test_split(ySF, test_size = .3, random_state = 42)

SFRegressionModel = LogisticRegression(max_iter=2000)
SFRegressionModel.fit(XTrain , yTrainSF)

print("\nThe accuracy of the SF model is:")
print(SFRegressionModel.score(XTest, yTestSF))

print("\nPredicted Values:")
print(SFRegressionModel.predict(XTest))
print("\nActual Values:")
print(yTestSF)

# PF Prediction
yPF = nbaplayerstats_df[['PF']]

yTrainPF, yTestPF = train_test_split(yPF, test_size = .3, random_state = 42)

PFRegressionModel = LogisticRegression(max_iter=2000)
PFRegressionModel.fit(XTrain, yTrainPF)

print("\nThe accuracy of the C model is:")
print(PFRegressionModel.score(XTest, yTestPF))

print("\nPredicted Values:")
print(PFRegressionModel.predict(XTest))
print("\nActual Values:")
print(yTestPF)


# C Prediction
yC = nbaplayerstats_df[['C']]

yTrainC, yTestC = train_test_split(yC, test_size = .3, random_state = 42)

CRegressionModel = LogisticRegression(max_iter=2000)
CRegressionModel.fit(XTrain, yTrainC)

print("\nThe accuracy of the C model is:")
print(CRegressionModel.score(XTest, yTestC))

print("\nPredicted Values:")
print(CRegressionModel.predict(XTest))
print("\nActual Values:")
print(yTestC)

''''
#Q1 How many observations and features in the 'iris' dataset?
print("Question #1:")
print("There are " + str(iris_df.shape[0]) + " observations")
print("There are " + str(iris_df.shape[1]) + " features")


# Split the data into training and test sets for linear regression (use a test set size of .3)
# Q2 Why (or why not) is .3 an appropriate size? Would you use something different?
trainingData, testData = train_test_split(iris_df, test_size = .3)
print("\nQuestion #2:")
print("Yes .3 is an appropriate size for Test Data, usually you want around a 70/30 split")


# Q3  Use the first three features to predict the fourth feature (sepal length, sepal width, and petal length
# to predict petal width) 
# What is the MSE and R-squared of the final model?  What do they mean?
print("\nQuestion #3:")

trainingDataX = trainingData[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']]
trainingDataY = trainingData[['petal width (cm)']]
testDataX = testData[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']]
testDataY = testData[['petal width (cm)']]

model = LinearRegression()
model.fit(trainingDataX, trainingDataY)
y_Predicted = model.predict(testDataX)

print("The mean squared error is " + str(mean_squared_error(testDataY, y_Predicted)))
print("The R^2 value is " + str(model.score(testDataX, testDataY)))



# Q4 Using the same three features to predict petal width,
# Compare three one-feature models and choose the best one based on MSE 
# Which model was chosen and why?  Graph the final model. (include a trend line)
print("\nQuestion 4:")

trainingDataY = trainingData[['petal width (cm)']]
testData1Y = testData[['petal width (cm)']]


trainingData1X = trainingData[['sepal length (cm)']]
testData1X = testData[['sepal length (cm)']]

trainingData2X = trainingData[['sepal width (cm)']]
testData2X = testData[['sepal width (cm)']]

trainingData3X = trainingData[['petal length (cm)']]
testData3X = testData[['petal length (cm)']]

model1 = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()

model1.fit(trainingData1X, trainingDataY)
model2.fit(trainingData2X, trainingDataY)
model3.fit(trainingData3X, trainingDataY)

y1_Predicted = model1.predict(testData1X)
y2_Predicted = model2.predict(testData2X)
y3_Predicted = model3.predict(testData3X)

print("The mean squared error of model 1 is " + str(mean_squared_error(testDataY, y1_Predicted)))
print("The mean squared error of model 2 is " + str(mean_squared_error(testDataY, y2_Predicted)))
print("The mean squared error of model 3 is " + str(mean_squared_error(testDataY, y3_Predicted)))
print("Model 3 is the best as it has the minimum MSE of the three models")

plt.scatter(trainingData1X, trainingDataY)
#plt.plot(model3, color='red', linestyle='--', linewidth=2, label='Trendline (Linear Regression)')
plt.show()

# Use  sepal length, sepal width, petal length, and petal width as IVs, and species as the DV for the following
# Subset the dataset to include only Setosa and Versicolor 
# Encoding for species: Setosa = 0, Versicolor = 1, Virginica = 2


# Split the data into training and test sets for logistic regression with test_size=0.3 and max_iter = 200
# Q5 What does the max_iter argument do in the logitic regression function?






# Q6 Calculate metrics for Logistic Regression, including the confusion matrix.  What do the metrics and  the confusion matrix
# indicate?  How likely would you expect to see theee types of metrics?
'''