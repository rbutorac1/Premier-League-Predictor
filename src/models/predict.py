import numpy as np
import pandas as pd

from scipy.stats import poisson
from xgboost import XGBRegressor
from datetime import date

TARGETS = [
        'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals', 'HomeTeamShots', 'AwayTeamShots',
        'HomeTeamShotsOnTarget', 'AwayTeamShotsOnTarget', 'HomeTeamCorners', 'AwayTeamCorners',
        'HomeTeamYellowCards', 'AwayTeamYellowCards', 'HomeTeamRedCards', 'AwayTeamRedCards'
    ]

FEATURE = [
    "FullTimeHomeTeamGoals_rolling_5",
    "FullTimeAwayTeamGoals_rolling_5",
    "HomeTeamShots_rolling_5",
    "AwayTeamShots_rolling_5",
    "HomeTeamShotsOnTarget_rolling_5",
    "AwayTeamShotsOnTarget_rolling_5",
    "HomeTeamCorners_rolling_5",
    "AwayTeamCorners_rolling_5",
    "HomeTeamYellowCards_rolling_5",
    "AwayTeamYellowCards_rolling_5",
    "HomeTeamRedCards_rolling_5",
    "AwayTeamRedCards_rolling_5",
    "Weekday",
    "Month",
    "IsWeekend",
    "ELO_home",
    "ELO_away",
    "ELO_diff"
]

FEATURE_MC = [
    "Weekday",
    "Month",
    "IsWeekend",
    "ELO_home",
    "ELO_away",
    "ELO_diff"
]

TARGET_MC = [
    "FullTimeHomeTeamGoals",
    "FullTimeAwayTeamGoals",
]


def load_model(target):
    DATA_PATH = f"data/models/{target}_model.json"
    model = XGBRegressor()
    model.load_model(DATA_PATH)
    return model

def match_input(home_team, away_team, match_date):
    match_date = pd.to_datetime(match_date)

    df = pd.read_csv("data/processed/PremierLeagueProcessed.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    home_hist = df[(df["HomeTeam"] == home_team) | (df["AwayTeam"] == home_team)]
    away_hist = df[(df["HomeTeam"] == away_team) | (df["AwayTeam"] == away_team)]

    home_hist = home_hist[home_hist["Date"] < match_date].sort_values("Date")
    away_hist = away_hist[away_hist["Date"] < match_date].sort_values("Date")

    if home_hist.empty or away_hist.empty:
        return

    home_last = home_hist.iloc[-1]
    away_last = away_hist.iloc[-1]

    row = {
        "Weekday": match_date.weekday(),
        "Month": match_date.month,
        "IsWeekend": 1 if match_date.weekday() in (5,6) else 0,
        "ELO_home": home_last["ELO_home"],
        "ELO_away": away_last["ELO_away"],
        "ELO_diff": home_last["ELO_home"] - away_last["ELO_away"],
    }

    for col in FEATURE:
        if "_rolling_" in col:
            if "Home" in col:
                row[col] = home_last[col]
            elif "Away" in col:
                row[col] = away_last[col]

    X = pd.DataFrame([row])
    X = X[FEATURE]
    return X




def predict_match(home_team, away_team, match_date):
    X = match_input(home_team, away_team, match_date)
    results = {}

    for target in TARGETS:
        model = load_model(target)
        prediction = model.predict(X)
        results[target] = float(prediction[0])

    lambda_H = results["FullTimeHomeTeamGoals"]
    lambda_A = results["FullTimeAwayTeamGoals"]

    MAX_GOALS = 9

    p_goals_H = [poisson.pmf(i, lambda_H) for i in range(MAX_GOALS)]
    p_goals_A = [poisson.pmf(i, lambda_A) for i in range(MAX_GOALS)]
    M = np.zeros((MAX_GOALS,MAX_GOALS))

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

def print_predictions(results):
    print("\n================ MATCH PREDICTION ================\n")

    print(">>> FINAL SCORE PREDICTION:")
    print(f"  Home Goals: {results['FullTimeHomeTeamGoals']:.2f}")
    print(f"  Away Goals: {results['FullTimeAwayTeamGoals']:.2f}\n")

    print(">>> MATCH EVENT PREDICTIONS:")

    for target in results:
        print(f"{target}: {results[target]:.2f}")

#print_predictions(predict_match("Sunderland", "Man City", date(2026, 1, 1)))

