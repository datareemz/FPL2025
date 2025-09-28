import sys
import pandas as pd
from api.FPL import FPL
from api.FPL_helpers import FPLHelpers
from auth.fpl_auth import FPLAuth
import aiohttp
import asyncio
from pycaret.classification import load_model, predict_model
from ml import ml
import logging

game_week = None
fixtures = None
players = None
MY_TEAM = []
metrics = [
    "roi",
    "points_per_min",
    "points_per_gw",
    "roi_per_gw",
    "goal_contributions_per_min",
    "roi_per_min",
    "total_points",
    "goals_scored",
    "assists",
    "minutes",
    "starts",
]

predictions_to_player_obj = []


# async def test():
#     session = aiohttp.ClientSession(trust_env=True)
#     auth = FPLAuth(session)
#     fpl = FPL(session, auth)
#     if fpl.logged_in():
#         print("Logged in for GW")
#         user = await fpl.get_user()
#         print("user:", user)
#         try:
#             my_team = await fpl.get_users_team(user)
#             print("my_team", my_team)
#         except Exception as err:
#             "Cant get users team info from FPL"
#         try:
#             my_players = await fpl.get_users_players(user)
#             print("my_players", my_players)
#
#             for i in my_players:
#                 player = await fpl.get_current_player(player_id=i['element'])
#                 print("player", player)
#
#             all_players = await fpl.get_all_current_players()
#
#             mean_roi = statistics.mean([x.roi() for x in all_players])
#             mean_points_per_min = statistics.mean([x.points_per_Min() for x in all_players])
#             player_pool = [ x for x in all_players if x.roi() > mean_roi and x.points_per_Min() > mean_points_per_min]
#
#             player_pool.sort(key=lambda x: x.roi(), reverse=True)
#
#             roi_model = load_model('ml/Final Lightgbm Model ROI-Target.pkl')
#             print("✅ Model loaded successfully!")
#
#             for i in player_pool:
#                 print(i)
#
#         except Exception as err:
#             "Cant get users players from FPL"
#         try:
#             game_week = await fpl.get_upcoming_gameweek()
#             print("next_gameweek", game_week)
#             fixtures = await fpl.get_fixtures_for_next_GW(int(game_week))
#             for i in fixtures:
#                 print(str(i))
#         except Exception as err:
#             "Cant get next gameweek from FPL"
#
#     await session.close()

# TODO
# async def get_historical_data():
#     try:
#         print("loading Model")
#         roi_model = load_model("../ml/Final Lightgbm Model ROI-Target")
#         print("ROI model loaded")
#     except Exception as e:
#         session = aiohttp.ClientSession(trust_env=True)
#         auth = FPLAuth(session)
#         helpers = FPLHelpers(session)
#         fpl = FPL(session, auth, helpers)
#         fpl.helpers.prepareData(historical=True)

# def get_current_data():
#     try:
#         print("loading Model")
#         roi_model = load_model("../ml/Final Lightgbm Model ROI-Target")
#         print("ROI model loaded")
#         session = aiohttp.ClientSession(trust_env=True)
#         auth = FPLAuth(session)
#         helpers = FPLHelpers(session)
#         fpl = FPL(session, auth, helpers)
#         fpl.helpers.prepareData(historical=False)
#     except Exception as e:
#         logging.error(e)


async def main():
    # roi_model = load_model("../ml/Final et Model ROI-Target")
    roi_model = load_model("ml/Final et Model ROI-Target")
    print("ROI model loaded")
    session = aiohttp.ClientSession(trust_env=True)
    auth = FPLAuth(session)
    helpers = FPLHelpers(session)
    fpl = FPL(session, auth, helpers)
    print("FPL model loaded")

    gameweek = await fpl.helpers.get_upcoming_gameweek()
    print(gameweek)
    #
    # await fpl.login()
    #
    # if fpl.logged_in():
    #     entry = await fpl.get_current_user_entry()
    #     print(entry)
    #     info = await fpl.get_manager_info()
    #     status = await fpl.get_transfers_status()
    #     print(status)
    #     print("Logged in")
    #     await session.close()

    # TODO retrain every 5 gameweeks
    print("Game Week: {}".format(gameweek))
    try:
        await asyncio.wait_for(fpl.helpers.getData(gameweek), timeout=60.0)
        unseen_data = pd.read_csv(f"datastore/current/FPL_data_{gameweek}.csv")

        print("\nPredictions on unseen data:")
        predictions = predict_model(roi_model, data=unseen_data).sort_values(
            "prediction_label", ascending=False
        )

        predictions = predictions.head(200)

        avg_ppg = predictions["points_per_game"].mean()
        avg_roi = predictions["roi"].mean()
        avg_proi = predictions["prediction_label"].mean()

        predictions = predictions[
            (predictions["points_per_game"] >= avg_ppg)
            & (predictions["roi"] >= avg_roi)
            & (predictions["prediction_label"] >= avg_proi)
        ]

        print(predictions.shape[0])
        predicted_players = predictions.to_dict("records")

        for i in predicted_players:
            convert = await fpl.get_current_player(player=i, convert_hist=False)
            predictions_to_player_obj.append(convert)

        print("\nPredicted player: ", predicted_players[-2])
        print("\nPredicted player converted: ", vars(predictions_to_player_obj[0]))

        await fpl.login()

        if fpl.logged_in():
            user = await fpl.get_user()
            my_players = await fpl.get_users_players(user)
            print("my_players", my_players)
            for i in my_players:
                player = await fpl.get_current_player(player_id=i["element"])
                MY_TEAM.append(player)
            MY_TEAM.sort(key=lambda x: x.roi())
            print("MY_TEAM", MY_TEAM[0])
            res = fpl.helpers.get_team_analysis(MY_TEAM, metrics=metrics)[
                "weakest_players"
            ]

            for i in res:
                candidates = fpl.helpers.find_valid_replacement(
                    player_out=i,
                    player_pool=predictions_to_player_obj,
                    current_team=MY_TEAM,
                    metrics=metrics,
                )
                print(i)
                print(len(predictions_to_player_obj))
                print(len(candidates))
                for i in candidates:
                    print(i["player"])
        await session.close()
    except Exception as err:
        "Cant get current data from FPL"
        print(err)
    gameweek = await fpl.helpers.get_gameweek_stats(5)
    print(gameweek)

    await fpl.login()

    if fpl.logged_in():
        entry = await fpl.get_current_user_entry()
        print(entry)
        info = await fpl.get_manager_info()
        status = await fpl.get_transfers_status()
        print(status)
        print("Logged in")
        await session.close()

    # TODO retrain every 5 gameweeks
    # print("Game Week: {}".format(gameweek))
    # try:
    #     await asyncio.wait_for(fpl.helpers.getData(gameweek), timeout=60.0)
    #     unseen_data = pd.read_csv(f"../datastore/current/FPL_data_{gameweek}.csv")
    #
    #     print("\nPredictions on unseen data:")
    #     predictions = predict_model(roi_model, data=unseen_data).sort_values(
    #         "prediction_label", ascending=False
    #     )
    #
    #     predictions = predictions.head(200)
    #
    #     avg_ppg = predictions["points_per_game"].mean()
    #     avg_roi = predictions["roi"].mean()
    #     avg_proi = predictions["prediction_label"].mean()
    #
    #     predictions = predictions[
    #         (predictions["points_per_game"] >= avg_ppg)
    #         & (predictions["roi"] >= avg_roi)
    #         & (predictions["prediction_label"] >= avg_proi)
    #     ]
    #
    #     print(predictions.shape[0])
    #     predicted_players = predictions.to_dict("records")
    #
    #     for i in predicted_players:
    #         convert = await fpl.get_current_player(player=i, convert_hist=False)
    #         predictions_to_player_obj.append(convert)
    #
    #     print("\nPredicted player: ", predicted_players[-2])
    #     print("\nPredicted player converted: ", vars(predictions_to_player_obj[0]))
    #
    #     await fpl.login()
    #
    #     if fpl.logged_in():
    #         user = await fpl.get_user()
    #         my_players = await fpl.get_users_players(user)
    #         print("my_players", my_players)
    #         for i in my_players:
    #             player = await fpl.get_current_player(player_id=i["element"])
    #             MY_TEAM.append(player)
    #         MY_TEAM.sort(key=lambda x: x.roi())
    #         print("MY_TEAM", MY_TEAM[0])
    #         res = fpl.helpers.get_team_analysis(MY_TEAM, metrics=metrics)[
    #             "weakest_players"
    #         ]
    #
    #         for i in res:
    #             candidates = fpl.helpers.find_valid_replacement(
    #                 player_out=i,
    #                 player_pool=predictions_to_player_obj,
    #                 current_team=MY_TEAM,
    #                 metrics=metrics,
    #             )
    #             print(i)
    #             print(len(predictions_to_player_obj))
    #             print(len(candidates))
    #             for i in candidates:
    #                 print(i["player"])
    #     await session.close()
    # except Exception as err:
    #     "Cant get current data from FPL"
    #     print(err)


asyncio.run(main())
