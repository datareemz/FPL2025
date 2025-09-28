"""
!!!!!! ONLY RUN ONCE !!!!!!!!!
** Requires Downloaded Files
** teams from : https://github.com/vaastav/Fantasy-Premier-League/blob/master/data/2024-25/teams.csv
** players from: https://github.com/vaastav/Fantasy-Premier-League/blob/master/data/2024-25/players_raw.csv

"""

import csv
import os
import requests
from prettytable import PrettyTable
from constants import POSITION_MAP, PLAYER_DATA_SCHEMA

url = "https://fantasy.premierleague.com/api/bootstrap-static/"
json_general = requests.get(url).json()
season = "24_25"
name_list = [
    str(player["first_name"] + " " + player["second_name"])
    for player in json_general["elements"]
]
player_id_list = [player["id"] for player in json_general["elements"]]
teamname_list = [team["name"] for team in json_general["teams"]]
teamid_list = [team["id"] for team in json_general["teams"]]
roi_list = []
teamid_list_g = []
pos_list_g = []
data_dict = []
id_dict = {}
team_dict = {}

with open(f"/docker/historical/_teams/teams_{season}.csv", mode="r") as teams:
    reader = csv.DictReader(teams)
    for team in reader:
        team_dict[int(team["id"])] = team["name"]
player_dict = {player_id_list[i]: name_list[i] for i in range(len(name_list))}
print(name_list)


# sorted_players = json_general["elements"]
players_table = PrettyTable()
players_table.field_names = [
    "Player_Name",
    "Price (Pounds)",
    "Games Played",
    "Goals",
    "Assists",
    "Direct Goal Contributions",
    "Total Points",
    "Points Per Game",
    "ROI",
    "Position",
    "Team",
    "Minutes",
]
players_table.align = "c"

with open(
    f"/docker/historical/_summary/players_raw_{season}.csv", mode="r"
) as sorted_players:
    print(team_dict)
    reader = csv.DictReader(sorted_players)
    headers = next(reader)
    for baller in reader:
        team = baller["team"]
        goals_scored = int(baller["goals_scored"])
        assists = int(baller["assists"])
        goal_contributions = goals_scored + assists
        starts = int(baller["starts"])
        minutes = int(baller["minutes"])
        teamname = team_dict[int((baller["team"]))]
        roi = float(
            f"{(int(baller['total_points']) / (int(baller['now_cost']) / 10)):0000000.4}"
        )
        roi_per_gw = roi / starts if starts > 0 else 0
        roi_list.append(roi)  # add all ROIs for all players

        pos = POSITION_MAP.get(int(baller["element_type"]), "MID")
        pos_list_g.append(pos)

        diction = PLAYER_DATA_SCHEMA.copy()
        diction.update(
            {
                "first_name": baller["first_name"],
                "second_name": baller["second_name"],
                "now_cost": round(int(baller["now_cost"]) / 10, 2),
                "starts": int(starts),
                "goals_scored": int(goals_scored),
                "assists": int(assists),
                "goal_contributions": int(goal_contributions),
                "total_points": int(baller["total_points"]),
                "points_per_game": float(baller["points_per_game"]),
                "roi": float(roi),
                "roi_per_gw": float(roi_per_gw),
                "element_type": baller["element_type"],
                "team": team,
                "minutes": int(minutes),
                "FDR_Average": 0.0,
                "team_name": teamname,
            }
        )

        data_dict.append(diction)
        csv_filename = f"FPL_data_{season}.csv"

        # Specify the target folder and filename
        target_folder = (
            "../historical/_summary"  # Replace with your desired folder name
        )

        # Create the target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        # Construct the full file path
        path = os.path.join(target_folder, csv_filename)

        keys = data_dict[0].keys()
        with open(path, mode="w", newline="") as file:
            dict_writer = csv.DictWriter(file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(data_dict)
