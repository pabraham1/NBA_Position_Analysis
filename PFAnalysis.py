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

# Print to ensure the program is working
print(nbaplayerstats_df[['Player', 'Pos_Encoded', 'Age', 'MP', 'G', 'PTS']])
print("Dataset Printed")



X = nbaplayerstats_df[['Age', 'G', 'GS', 'MP', 'PTS', 'AST', 'ORB', 'DRB', 'STL', 'BLK', 'TOV', 'PF',
                        '3P', '3PA', '2P', '2PA', 'FG', 'FGA', 'FT', 'FTA']]
yPF = nbaplayerstats_df[['C']]

XTrain, XTest, yTrainPF, yTestPF = train_test_split(X, yPF, test_size = .3, random_state = 42)

PFRegressionModel = LogisticRegression(max_iter=2000)
PFRegressionModel.fit(XTrain, yTrainPF)

print("\nThe accuracy of the C model is:")
print(PFRegressionModel.score(XTest, yTestPF))

print("\nPredicted Values:")
print(PFRegressionModel.predict(XTest))
print("\nActual Values:")
print(yTestPF)