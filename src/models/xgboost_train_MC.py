import numpy as np
import pandas as pd

from scipy.stats import poisson
from xgboost import XGBRegressor
from datetime import date

from src.models.predict import TARGET_MC

FEATURES_MC = [
    "Weekday",
    "Month",
    "IsWeekend",
    "ELO_home",
    "ELO_away",
    "ELO_diff"
]

TARGETS_MC = [
    "FullTimeHomeTeamGoals_MC",
    "FullTimeAwayTeamGoals_MC"
]

def dataset_MC():
    df = pd.read_csv("data/processed/PremierLeagueProcessed.csv")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Weekday"] = df["Date"].dt.weekday
    df["Month"] = df["Date"].dt.month
    df["IsWeekend"] = df["Weekday"].isin([5, 6]).astype(int)

    df["ELO_diff"] = df["ELO_home"] - df["ELO_away"]

    X = df[FEATURES_MC]
    y_home = df["FullTimeHomeTeamGoals"]
    y_away = df["FullTimeAwayTeamGoals"]

    return X, y_home, y_away

def train_MC():
    X, y_home, y_away = dataset_MC()

    home_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    away_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    home_model.fit(X, y_home)
    away_model.fit(X, y_away)

    home_model.get_booster().save_model("data/models/models_MC/FullTimeHomeTeamGoals_MC.json")
    away_model.get_booster().save_model("data/models/models_MC/FullTimeAwayTeamGoals_MC.json")

if __name__ == "__main__":
    train_MC()