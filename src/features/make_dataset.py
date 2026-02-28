import pandas as pd
import numpy as np

from src.features.ELO_rating import add_ELO
from src.preprocessing.data_preparation import rolling_features

# values that are going to be predicted
TARGETS = [
        'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals', 'HomeTeamShots', 'AwayTeamShots',
        'HomeTeamShotsOnTarget', 'AwayTeamShotsOnTarget', 'HomeTeamCorners', 'AwayTeamCorners',
        'HomeTeamYellowCards', 'AwayTeamYellowCards', 'HomeTeamRedCards', 'AwayTeamRedCards'
    ]

def load_data(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    return df

# X features
def feature_columns(df):
    rolling_columns = [column for column in df.columns if "_rolling_" in column]
    context_columns = ["Weekday", "Month", "IsWeekend", "ELO_home", "ELO_away", "ELO_diff"]
    X_features = rolling_columns + context_columns

    return X_features

# Returns dictionary: datasets[target] = {"X": X, "y": y}
def create_Xy(df, X_features):
    datasets = {}

    for target in TARGETS:
        y = df[target]
        X = df[X_features]

        datasets[target] = {
            "X": X,
            "y": y
        }

    return datasets

def split_by_season(df, X, y, train_until):
    df["SeasonStart"] = df["Season"].str[:4].astype(int)

    train = df["SeasonStart"] <= train_until
    test = df["SeasonStart"] > train_until

    X_train = X[train]
    y_train = y[train]

    X_test = X[test]
    y_test = y[test]

    return X_train, X_test, y_train, y_test

def make_dataset():
    DATA_PATH = "data/processed/PremierLeagueProcessed.csv"

    df = load_data(DATA_PATH)
    df = rolling_features(df, 5)
    df = add_ELO(df)
    X_features = feature_columns(df)
    datasets = create_Xy(df, X_features)



    for target in TARGETS:
        X = datasets[target]["X"]
        y = datasets[target]["y"]

        X_train, X_test, y_train, y_test = split_by_season(df, X, y, 2025)

        datasets[target] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

    print("Dataset made successfully")

    return datasets
