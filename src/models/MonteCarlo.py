import numpy as np
import pandas as pd

from predict_MC import predict_match_MC

num_simulations = 10000

class Team:
    def __init__(self, name):
        self.name = name
        self.points = 0
        self.ELO = 0
        self.winner = 0
        self.top4 = 0
        self.relegation = 0
        self.averagePos = 0
        self.avgPts = 0

    def addPts(self, pts):
        self.points += pts

    def resetPts(self):
        self.points = 0

    def setELO(self, elo):
        self.ELO = elo

Teams = []
Teams_dict = {}
idx_Teams = {}

Arsenal = Team("Arsenal")
Teams.append(Arsenal)
Teams_dict["Arsenal"] = Arsenal
idx_Teams["Arsenal"] = 0

ManCity = Team("Man City")
Teams.append(ManCity)
Teams_dict["Man City"] = ManCity
idx_Teams["Man City"] = 1

ManUnited = Team("Man United")
Teams.append(ManUnited)
Teams_dict["Man United"] = ManUnited
idx_Teams["Man United"] = 2

Chelsea = Team("Chelsea")
Teams.append(Chelsea)
Teams_dict["Chelsea"] = Chelsea
idx_Teams["Chelsea"] = 3

Tottenham = Team("Tottenham")
Teams.append(Tottenham)
Teams_dict["Tottenham"] = Tottenham
idx_Teams["Tottenham"] = 4

WestHam = Team("West Ham")
Teams.append(WestHam)
Teams_dict["West Ham"] = WestHam
idx_Teams["West Ham"] = 5

AstonVilla = Team("Aston Villa")
Teams.append(AstonVilla)
Teams_dict["Aston Villa"] = AstonVilla
idx_Teams["Aston Villa"] = 6

Liverpool = Team("Liverpool")
Teams.append(Liverpool)
Teams_dict["Liverpool"] = Liverpool
idx_Teams["Liverpool"] = 7

Brentford = Team("Brentford")
Teams.append(Brentford)
Teams_dict["Brentford"] = Brentford
idx_Teams["Brentford"] = 8

Bournemouth = Team("Bournemouth")
Teams.append(Bournemouth)
Teams_dict["Bournemouth"] = Bournemouth
idx_Teams["Bournemouth"] = 9

Everton = Team("Everton")
Teams.append(Everton)
Teams_dict["Everton"] = Everton
idx_Teams["Everton"] = 10

Fulham = Team("Fulham")
Teams.append(Fulham)
Teams_dict["Fulham"] = Fulham
idx_Teams["Fulham"] = 11

Sunderland = Team("Sunderland")
Teams.append(Sunderland)
Teams_dict["Sunderland"] = Sunderland
idx_Teams["Sunderland"] = 12

CrystalPalace = Team("Crystal Palace")
Teams.append(CrystalPalace)
Teams_dict["Crystal Palace"] = CrystalPalace
idx_Teams["Crystal Palace"] = 13

Leeds = Team("Leeds")
Teams.append(Leeds)
Teams_dict["Leeds"] = Leeds
idx_Teams["Leeds"] = 14

Newcastle = Team("Newcastle")
Teams.append(Newcastle)
Teams_dict["Newcastle"] = Newcastle
idx_Teams["Newcastle"] = 15

NottinghamForest = Team("Nott'm Forest")
Teams.append(NottinghamForest)
Teams_dict["Nott'm Forest"] = NottinghamForest
idx_Teams["Nott'm Forest"] = 16

Burnley = Team("Burnley")
Teams.append(Burnley)
Teams_dict["Burnley"] = Burnley
idx_Teams["Burnley"] = 17

Wolves = Team("Wolves")
Teams.append(Wolves)
Teams_dict["Wolves"] = Wolves
idx_Teams["Wolves"] = 18

Brighton = Team("Brighton")
Teams.append(Brighton)
Teams_dict["Brighton"] = Brighton
idx_Teams["Brighton"] = 19

def startELO():
    df = pd.read_csv("data/processed/PremierLeagueProcessed.csv")
    df["SeasonStart"] = df["Season"].str[:4].astype(int)
    lastELO_df = df[(df["SeasonStart"] == 2024) & (df["MatchWeek"] == 38)]
    for index, row in lastELO_df.iterrows():
        team1 = row["HomeTeam"]
        team2 = row["AwayTeam"]
        team1ELO = row["ELO_home"]
        team2ELO = row["ELO_away"]
        for team in Teams:
            if team.name == team1:
                team.setELO(team1ELO)
            if team.name == team2:
                team.setELO(team2ELO)

    elo_sum = 0.0
    for team in Teams:
        elo_sum += team.ELO
    elo_promoted = (elo_sum / 17.0) - 50

    for team in Teams:
        if team.ELO == 0:
            team.setELO(elo_promoted)

def reset_season():
    startELO()
    for team in Teams:
        team.resetPts()

def simulate_season():
    reset_season()
    fixtures = pd.read_csv("data/processed/premier_league_2025_2026_fixtures.csv")

    for index, row in fixtures.iterrows():
        date = row["date"]
        HomeTeam = row["home"]
        AwayTeam = row["away"]

        result = predict_match_MC(Teams_dict[HomeTeam], Teams_dict[AwayTeam], date)

        p_H = result["ProbabilityHomeWin"]
        p_A = result["ProbabilityAwayWin"]
        p_D = result["ProbabilityDraw"]

        probability = np.array([p_H, p_A, p_D])
        probability = probability / probability.sum()

        outcome = np.random.choice(["H", "A", "D"], p=probability)

        real_oc_H = 0
        real_oc_A = 0

        if outcome == "H":
            real_oc_H = 1
            Teams_dict[HomeTeam].addPts(3)
        elif outcome == "A":
            real_oc_A = 1
            Teams_dict[AwayTeam].addPts(3)
        else:
            real_oc_H = 0.5
            real_oc_A = 0.5
            Teams_dict[HomeTeam].addPts(1)
            Teams_dict[AwayTeam].addPts(1)

        K = 30

        home_elo = Teams_dict[HomeTeam].ELO
        away_elo = Teams_dict[AwayTeam].ELO

        exp_home = 1.0 / (1 + 10 ** ((away_elo - (home_elo + 50)) / 400))
        exp_away = 1.0 - exp_home

        new_home = home_elo + K *(real_oc_H - exp_home)
        new_away = away_elo + K *(real_oc_A - exp_away)

        Teams_dict[HomeTeam].setELO(new_home)
        Teams_dict[AwayTeam].setELO(new_away)

def monte_carlo():
    position_matrix = np.zeros((20,20))

    for i in range(num_simulations):
        reset_season()
        simulate_season()
        table = sorted(Teams_dict.values(), key=lambda team: team.points, reverse=True)
        for team in table:
            team.avgPts += team.points
            team_idx = idx_Teams[team.name]
            for j in range(len(table)):
                if team.name == table[j].name:
                    position = j
                    position_matrix[team_idx][position] += 1
                    break

    position_matrix /= (num_simulations * 1.0)

    for team in Teams:
        team_idx = idx_Teams[team.name]
        team.avgPts /= (num_simulations * 1.0)

        team_positions = position_matrix[team_idx]
        pos_vector = np.arange(1, 21)

        team.winner = position_matrix[team_idx][0]
        team.averagePos = team_positions @ pos_vector
        team.top4 = position_matrix[team_idx][0] + position_matrix[team_idx][1] + position_matrix[team_idx][2] + position_matrix[team_idx][3]
        team.relegation = position_matrix[team_idx][17] + position_matrix[team_idx][18] + position_matrix[team_idx][19]

monte_carlo()
table = sorted(Teams_dict.values(), key=lambda team: team.winner, reverse=True)
for team in table:
    print(f"{team.name:<20}: TITLE: {team.winner*100:>8.2f}% | "
          f"TOP4: {team.top4*100:>8.2f}% | "
          f"RELEGATION: {team.relegation*100:>8.2f}% | "
          f"AVG POSITION: {team.averagePos:>8.2f} | "
          f"AVG POINTS{team.avgPts:>8.2f} |")




