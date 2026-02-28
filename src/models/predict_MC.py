import numpy as np
import pandas as pd

from scipy.stats import poisson
from xgboost import XGBRegressor
from datetime import date

FEATURES_MC = [
    "Weekday",
    "Month",
    "IsWeekend",
    "ELO_home",
    "ELO_away",
    "ELO_diff"
]

def load_model(target):
    DATA_PATH = f"data/models/models_MC/{target}.json"
    model = XGBRegressor()
    model.load_model(DATA_PATH)
    return model

def match_input_MC(homeTeam, awayTeam, match_date):
    match_date = pd.to_datetime(match_date)

    row = {
        "Weekday": match_date.weekday(),
        "Month": match_date.month,
        "IsWeekend": 1 if match_date.weekday() in (5,6) else 0,
        "ELO_home": homeTeam.ELO,
        "ELO_away": awayTeam.ELO,
        "ELO_diff": homeTeam.ELO - awayTeam.ELO
    }

    X = pd.DataFrame([row])
    return X[FEATURES_MC]

model_home_MC = load_model("FullTimeHomeTeamGoals_MC")
model_away_MC = load_model("FullTimeAwayTeamGoals_MC")

def predict_match_MC(homeTeam, awayTeam, match_date):
    X = match_input_MC(homeTeam, awayTeam, match_date)
    results = {}

    prediction = model_home_MC.predict(X)
    results["FullTimeHomeTeamGoals_MC"] = float(prediction[0])

    prediction = model_away_MC.predict(X)
    results["FullTimeAwayTeamGoals_MC"] = float(prediction[0])

    lambda_H = results["FullTimeHomeTeamGoals_MC"]
    lambda_A = results["FullTimeAwayTeamGoals_MC"]

    MAX_GOALS = 9

    p_goals_H = [poisson.pmf(i, lambda_H) for i in range(MAX_GOALS)]
    p_goals_A = [poisson.pmf(i, lambda_A) for i in range(MAX_GOALS)]
    M = np.zeros((MAX_GOALS, MAX_GOALS))

    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            M[i][j] = p_goals_H[i] * p_goals_A[j]

    p_HomeWin = 0
    p_AwayWin = 0
    p_Draw = 0

    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            if i == j:
                p_Draw += M[i][j]
            elif i > j:
                p_HomeWin += M[i][j]
            else:
                p_AwayWin += M[i][j]

    results["ProbabilityHomeWin"] = p_HomeWin
    results["ProbabilityAwayWin"] = p_AwayWin
    results["ProbabilityDraw"] = p_Draw

    return results