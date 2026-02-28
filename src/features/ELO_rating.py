import pandas as pd
import numpy as np

teams = {}

def ELO_columns(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df["ELO_home"] = 1600.0
    df["ELO_away"] = 1600.0
    return df

def get_ELO(team):
    return teams.get(team, 1600)  #ako ne postoji key(team) vraca 1600

def set_ELO(team, value):
    teams[team] = value

def calculate_ELO(df):
    df = df.sort_values("Date").reset_index(drop=True)
    K = 30.0

    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    elo = {team: 1600 for team in teams}

    df["ELO_home"] = 0.0
    df["ELO_away"] = 0.0

    for i, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        home_elo = elo[home]
        away_elo = elo[away]

        df.at[i, "ELO_home"] = home_elo
        df.at[i, "ELO_away"] = away_elo

        exp_home = 1.0 / (1 + 10 ** ((away_elo - (home_elo + 50)) / 400))
        exp_away = 1.0 - exp_home

        if row["FullTimeResult"] == "H":
            s_home, s_away = 1.0, 0
        elif row["FullTimeResult"] == "A":
            s_home, s_away = 0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        elo[home] = home_elo + K * (s_home - exp_home)
        elo[away] = away_elo + K * (s_away - exp_away)

    return df


def add_ELO(df):
    df = calculate_ELO(df)
    df["ELO_diff"] = df["ELO_home"] - df["ELO_away"]
    return df
