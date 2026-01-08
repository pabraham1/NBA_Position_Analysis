import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the NBA Player Stats dataset
nbaplayerstats_df = pd.read_csv('nba_dataset(NBA).csv')

# Filter out a player's season stats, who played less than a certain threshold of minutes in that season
nbaplayerstats_df = nbaplayerstats_df[nbaplayerstats_df['MP'] * nbaplayerstats_df['G'] > 500]

print(nbaplayerstats_df[nbaplayerstats_df['Player'] == 'Anthony Miller'])

# Label Encode the Positions
le = LabelEncoder()
nbaplayerstats_df['Pos_Encoded'] = le.fit_transform(nbaplayerstats_df['Pos'])


'''
# Print to ensure the program is working
print(nbaplayerstats_df[['Player', 'Pos_Encoded', 'Age', 'MP', 'G', 'PTS']])
print("Dataset Printed")
'''


# Multi Position Prediction
#X = nbaplayerstats_df[['Age', 'G', 'GS', 'MP', 'PTS', 'AST', 'ORB', 'DRB', 'STL', 'BLK', 'TOV', 'PF',
#                        '3P', '3PA', '2P', '2PA', 'FG', 'FGA', 'FT', 'FTA']]
X = nbaplayerstats_df[['Age', 'GS', 'MP', 'AST', 'ORB', 'DRB', 'STL', 'BLK', 'TOV', 'PF',
                        '3P', '3PA', '2P', '2PA', 'FG', 'FGA', 'FT', 'FTA']]
y = nbaplayerstats_df['Pos_Encoded']

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = .3, random_state = 42)

# Scaling the inputs
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)

RegressionModel = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', max_iter = 2000)
RegressionModel.fit(XTrain , yTrain)

print("\nThe accuracy of the Multi-Position model is:")
print(RegressionModel.score(XTest, yTest))

print("\nPredicted Values:")
print(RegressionModel.predict(XTest))
print("\nActual Values:")
print(yTest)
