import pandas as pd
from api.FPL import FPL
from api.FPL_helpers import FPLHelpers
from auth.fpl_auth import FPLAuth
import aiohttp
import asyncio
from pycaret.regression import (
    load_model,
    predict_model,
    setup,
    tune_model,
    finalize_model,
    save_model,
    create_model,
)
import logging

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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
    roi_model = load_model("ml/Final et Model ROI-Target - Retrained")
    logger.info("ROI model loaded")
    session = aiohttp.ClientSession(trust_env=True)
    auth = FPLAuth(session)
    helpers = FPLHelpers(session)
    fpl = FPL(session, auth, helpers)
    await fpl.login()

    # historical = pd.read_csv(f"datastore/training/FPL_data_24_25.csv")
    # new_data = pd.read_csv(f"datastore/current/FPL_data_{prev_gameweek}.csv")
    # print(historical.shape)
    # print(new_data.shape)
    #
    # # s = setup(data=new_data, target='roi', session_id=123)
    # # updated_model = create_model(roi_model)
    #
    # combined_data = pd.concat([historical, new_data], ignore_index=True)
    #
    #
    # print(combined_data.shape)
    # combined_data = combined_data[(combined_data["starts"] >  0 )]
    # print('-------------------------')
    # print(combined_data.shape)
    # print(combined_data.head(10))
    #
    #
    # data = combined_data.sample(frac=0.80, random_state=786)
    # data_unseen = combined_data.drop(data.index)
    #
    # data.reset_index(drop=True, inplace=True)
    # data_unseen.reset_index(drop=True, inplace=True)
    #
    # # Debug: Check sampled data
    # print('Data for modeling shape:', data.shape)
    # print('ROI in sampled data - unique values:', data['roi'].nunique())
    #
    #
    # s = setup(data=data, target='roi', session_id=7177,ignore_features=['first_name', 'second_name', 'team_name', 'team', 'element_type'])
    # et_retrain = create_model('et')
    # tuned_et = tune_model(et_retrain)
    # final_retrained_model = finalize_model(tuned_et)
    #
    # save_model(final_retrained_model, 'Final et Model ROI-Target - Retrained')
    # print(str(final_retrained_model))

    # status = await fpl.get_transfers_status()
    # stats = await fpl.helpers.get_gameweek_stats(gw=prev_gameweek)
    # print(status)
    # print(info)
    # print(stats)
    gameweek = await fpl.helpers.get_upcoming_gameweek()
    prev_gameweek = gameweek - 1

    # TODO retrain every 5 gameweeks
    logger.info("upcoming Game Week: {}".format(gameweek))
    try:
        await asyncio.wait_for(fpl.helpers.getData(gameweek), timeout=60.0)
        # unseen_data = pd.read_csv(f"../datastore/current/FPL_data_{gameweek}.csv")
        unseen_data = pd.read_csv(f"datastore/current/FPL_data_{gameweek}.csv")
        print(unseen_data.head(10))
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

        logger.info("Predictions pool on unseen data: %d", predictions.shape[0])
        predicted_players = predictions.to_dict("records")

        for i in predicted_players:
            convert = await fpl.get_current_player(player=i, convert_hist=False)
            predictions_to_player_obj.append(convert)

        logger.info("upcoming Game Week: {}".format(gameweek))
        info = await fpl.get_manager_info_for_gw(gw=prev_gameweek)
        logger.info(
            "(previous) Game Week {} Manager info: {}".format(prev_gameweek, info)
        )
        gw_stats = await fpl.get_gameweek_stats(gw=prev_gameweek)
        logger.info("(previous) Game Week {} Stats: {}".format(prev_gameweek, gw_stats))
        logger.info(
            "U_Highest {}".format(abs(info["points"] - gw_stats["highest_score"]))
        )
        logger.info(
            "U_Average {}".format(abs(info["points"] - gw_stats["average_entry_score"]))
        )

        if fpl.logged_in():
            user = await fpl.get_user()
            my_players = await fpl.get_users_players(user)
            for i in my_players:
                player = await fpl.get_current_player(player_id=i["element"])
                MY_TEAM.append(player)
            MY_TEAM.sort(key=lambda x: x.roi())
            res = fpl.helpers.get_team_analysis(MY_TEAM, metrics=metrics)
            res = res["no_position"]
            print(res)

            for i in res[0:3]:
                candidates = fpl.helpers.find_valid_replacement(
                    player_out=i,
                    player_pool=predictions_to_player_obj,
                    current_team=MY_TEAM,
                    metrics=metrics,
                )
                logger.info("Weak player: %s", i)
                logger.info("Number of potential candidates: %d", len(candidates))
                for i in candidates:
                    logger.info("Candidate: %s", i["player"])
        # entry = await fpl.get_current_user_entry()
        # print(entry)
        # info = await fpl.get_manager_info()
        # status = await fpl.get_transfers_status()
        # print(status)
        # print(info)
        logger.info("Done work flow")
        await session.close()
    except Exception as err:
        "Cant get current data from FPL"
        logger.error("Error in main workflow: %s", err)
    await session.close()


asyncio.run(main())
