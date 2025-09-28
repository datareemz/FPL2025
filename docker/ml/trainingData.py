import pandas as pd

season = "24_25"  # HardCoded for now
from historical import historical

csv_file_path = f"../historical/_summary/FPL_data_{season}.csv"
csv_file_path_dest = f"../datastore/training/FPL_data_{season}.csv"

form_dict, r = historical.getFormDict(season=season)
print(form_dict)
FDR = historical.get_FDR(form_dict, season=season)
print(FDR)
# Assumed promoted will get relegated at the start of the season
FDR["Burnley"] = FDR["Ipswich"]
FDR["Sunderland"] = FDR["Leicester"]
FDR["Leeds"] = FDR["Southampton"]
df = pd.read_csv(csv_file_path)


df["FDR_Average"] = df.apply(lambda x: FDR[x["team_name"]], axis=1)
df.to_csv(csv_file_path_dest, index=False)
print("done")
