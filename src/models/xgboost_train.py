import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from src.features.make_dataset import make_dataset

TARGETS = [
        'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals', 'HomeTeamShots', 'AwayTeamShots',
        'HomeTeamShotsOnTarget', 'AwayTeamShotsOnTarget', 'HomeTeamCorners', 'AwayTeamCorners',
        'HomeTeamYellowCards', 'AwayTeamYellowCards', 'HomeTeamRedCards', 'AwayTeamRedCards'
    ]

datasets = make_dataset()

for target in TARGETS:
    X_train = datasets[target]["X_train"]
    y_train = datasets[target]["y_train"]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,  # percentage of rows seen by each tree (the best generalization)
        colsample_bytree=0.8,  # percentage of features seen by each tree (reduces correlation)
        random_state=42,
    )

    model.fit(X_train, y_train)

    booster = model.get_booster()
    booster.save_model(f"data/models/{target}_model.json")

    print(f"{target} model saved.")

print("\nTraining complete.")





