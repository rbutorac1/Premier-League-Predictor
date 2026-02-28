import pandas as pd
from math import exp, factorial
from datetime import datetime
from predict import predict_match


def poisson_pmf(l, k):
    return (l**k * exp(-l)) / factorial(k)

def poisson_probs(hg, ag, max_g=8):
    pH = pD = pA = 0
    for i in range(max_g):
        for j in range(max_g):
            p = poisson_pmf(hg, i) * poisson_pmf(ag, j)
            if i > j:
                pH += p
            elif i == j:
                pD += p
            else:
                pA += p
    return pH, pD, pA


def test():
    df = pd.read_csv("data/processed/PremierLeagueProcessed.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["SeasonStart"] = df["Season"].str[:4].astype(int)

    df_test = df[df["SeasonStart"] >= 2020].copy()

    mae_home = []
    mae_away = []
    brier_scores = []
    acc = []

    for _, row in df_test.iterrows():

        home = row["HomeTeam"]
        away = row["AwayTeam"]
        date = row["Date"]

        pred = predict_match(home, away, date)

        ph = pred["FullTimeHomeTeamGoals"]
        pa = pred["FullTimeAwayTeamGoals"]

        mae_home.append(abs(ph - row["FullTimeHomeTeamGoals"]))
        mae_away.append(abs(pa - row["FullTimeAwayTeamGoals"]))

        pH, pD, pA = poisson_probs(ph, pa)

        actual = row["FullTimeResult"]
        onehot = (1,0,0) if actual=="H" else (0,1,0) if actual=="D" else (0,0,1)

        brier = ((pH - onehot[0])**2 +
                 (pD - onehot[1])**2 +
                 (pA - onehot[2])**2)
        brier_scores.append(brier)

        pred_label = "H" if pH>=max(pH,pD,pA) else ("D" if pD>=max(pH,pD,pA) else "A")
        acc.append(pred_label == actual)

    print("\n===== TEST RESULTS =====")
    print(f"MAE Home Goals : {sum(mae_home)/len(mae_home):.4f}")
    print(f"MAE Away Goals : {sum(mae_away)/len(mae_away):.4f}")
    print(f"Brier Score    : {sum(brier_scores)/len(brier_scores):.4f}")
    print(f"Accuracy       : {sum(acc)/len(acc):.4f}")
    print("=================================\n")


if __name__ == "__main__":
    test()
