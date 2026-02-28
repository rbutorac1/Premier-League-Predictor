from operator import ifloordiv

import pandas as pd
import numpy as np

from src.features.ELO_rating import add_ELO


def load_raw_data(path):
    df = pd.read_csv(path)
    return df



def remove_columns(df):
    drop_columns = [column for column in df.columns
                    if column.startswith('B365') or column.startswith('Market')]
    df = df.drop(columns=drop_columns)
    return df



def normalize_columns(df):
    df['HomeTeam'] = df['HomeTeam'].str.strip()  # removing whitespaces
    df['AwayTeam'] = df['AwayTeam'].str.strip()

    df['HomeTeam'] = df['HomeTeam'].str.title()
    df['AwayTeam'] = df['AwayTeam'].str.title()

    df['Date'] = pd.to_datetime(df['Date']) # date to datetime format

    df['Weekday'] = df['Date'].dt.weekday # dow between 0 and 6
    df['Month'] = df['Date'].dt.month # extracting month
    df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)

    df.sort_values(by=['Season', 'MatchWeek', 'Date'], inplace = True)
    df.reset_index(drop = True, inplace = True)

    return df



def rolling_features(df, window=5):
    df.sort_values(by=['Season', 'MatchWeek', 'Date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    stats = [
        'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals', 'HomeTeamShots', 'AwayTeamShots','HomeTeamShotsOnTarget',
        'AwayTeamShotsOnTarget', 'HomeTeamCorners', 'AwayTeamCorners', 'HomeTeamYellowCards', 'AwayTeamYellowCards',
        'HomeTeamRedCards', 'AwayTeamRedCards'
    ]

    for stat in stats:
        team_col = "HomeTeam" if "Home" in stat else "AwayTeam"

        df[f"{stat}_rolling_{window}"] = (df.groupby(team_col)[stat].transform(lambda x: x.shift(1).rolling(window).mean()))

    return df




def save_data(df, output_path):
    df.to_csv(output_path, index=False)



def prepare_data(input_path, output_path):
    df = load_raw_data(input_path)
    df = remove_columns(df)
    df = normalize_columns(df)
    df = rolling_features(df, window=5)
    df = add_ELO(df)

    df["SeasonStart"] = df["Season"].str[:4].astype(int)
    df= df[df["SeasonStart"] > 2000]
    df = df.drop(columns=["SeasonStart"])



    save_data(df, output_path)

prepare_data("data/raw/PremierLeague.csv"
             , "data/processed/PremierLeagueProcessed.csv")


