import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

# Takes the input dataframe and runs a Multinomial Logistic Regression on the input dataframe
# i.e., tries to predict the position of each player in the test set after training on the training set
def MultiLogRegRunner(df_input, title):
    # Define Inputs and Outputs
    X = df_input[['Age', 'G', 'GS', 'MP', 'PTS', 'AST', 'ORB', 'DRB', 'STL', 'BLK', 'TOV', 'PF',
                            '3P', '3PA', '2P', '2PA', 'FG', 'FGA', 'FT', 'FTA']]
    y = df_input['Pos_Encoded']

    # random_state = 42
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = .3)

    # Scaling the inputs
    sc = StandardScaler()
    XTrain = sc.fit_transform(XTrain)
    XTest = sc.transform(XTest)

    # create the LogReg model, using method 'solver' and stopping at max_iter if it doesn't converge
    RegressionModel = LogisticRegression(solver = 'lbfgs', max_iter = 200)
    RegressionModel.fit(XTrain, yTrain)

    print("\nThe accuracy of the " + title + " Multi-Position model is:")
    print(RegressionModel.score(XTest, yTest))

    cm = confusion_matrix(yTest, RegressionModel.predict(XTest), labels = [1,2,3,4,5])
    confMatDisp = ConfusionMatrixDisplay(cm, display_labels=['PG', 'SG', 'SF', 'PF', 'C'])
    confMatDisp.plot()
    plt.title(title)
    #plt.show()

    '''
    print("\nPredicted Values:")
    print(RegressionModel.predict(XTest))
    print("\nActual Values:")
    print(yTest)
    '''
    

# Load the NBA Player Stats dataset
nbaplayerstats_df = pd.read_csv('nba_dataset(NBA).csv')

# Filter out player's-seasons that had 500 or less minutes played on the season
nbaplayerstats_df = nbaplayerstats_df[nbaplayerstats_df['MP'] * nbaplayerstats_df['G'] > 500]

# Label Encode the Positions
# Add one to all the encodings so that PG starts at 1 instead of 0, C is 5 instead of 4
oe = OrdinalEncoder(categories = [['PG', 'SG', 'SF', 'PF', 'C']])
nbaplayerstats_df['Pos_Encoded'] = oe.fit_transform(nbaplayerstats_df[['Pos']])
nbaplayerstats_df['Pos_Encoded'] = nbaplayerstats_df[['Pos_Encoded']] + 1


# CREATING DATAFRAMES
# Creating decades dataframes
nba80sstats_df = nbaplayerstats_df[nbaplayerstats_df['Season'] < '1990-1991']

nba90sstats_df = nbaplayerstats_df[nbaplayerstats_df['Season'] >= '1990-1991']
nba90sstats_df = nba90sstats_df[nba90sstats_df['Season'] < '2000-2001']

nba00sstats_df = nbaplayerstats_df[nbaplayerstats_df['Season'] >= '2000-2001']
nba00sstats_df = nba00sstats_df[nba00sstats_df['Season'] < '2010-2011']

# The 20s data are lumped in with the 10s since there are only 5 seasons in the 20s so far
nba10sstats_df = nbaplayerstats_df[nbaplayerstats_df['Season'] >= '2010-2011']

# Creating 20ish-year dataframes
nba8090sstats_df = nbaplayerstats_df[nbaplayerstats_df['Season'] < '2000-2001']
nba0010sstats_df = nbaplayerstats_df[nbaplayerstats_df['Season'] >= '2000-2001']


# CREATING MODELS
# Creating Decade Models
MultiLogRegRunner(nba80sstats_df, '1980s Model')
MultiLogRegRunner(nba90sstats_df, '1990s Model')
MultiLogRegRunner(nba00sstats_df, '2000s Model')
MultiLogRegRunner(nba10sstats_df, '2010s Model')

# Creating 20ish-year Models
MultiLogRegRunner(nba8090sstats_df, 'Pre-2000s Model')
MultiLogRegRunner(nba0010sstats_df, 'Post-2000s Model')

# Creating 45-year Model (i.e. using all the data)
MultiLogRegRunner(nbaplayerstats_df, 'All-Time Model')

plt.show()


