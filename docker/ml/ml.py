# import csv
# from api.FPL_helpers import FPLHelpers
# import aiohttp
#
#
# async def getData():
#     session = aiohttp.ClientSession(trust_env=True)
#     helpers = FPLHelpers(session)
#     next_gw = await helpers.get_upcoming_gameweek()
#     print("Next GW: {}".format(next_gw))
#
#
#     fixtures = await helpers.get_all_fixtures(*range(1, next_gw))
#     f, s = helpers.getFormDict(fixtures=fixtures)
#     g = helpers.get_FDR(form_dict=f, fixtures=fixtures)
#     dict = await helpers.prepareData()
#     for _ in dict:
#         if next_gw == 1:
#             _["games_played"] = 0
#             _["minutes"] = 0
#         if _["team"] in g.keys():
#             _["FDR_Average"] = round(g[_["team"]], 3)
#     keys = dict[0].keys()
#     with open(
#         f"../datastore/current/FPL_data_{next_gw}.csv", "w", newline=""
#     ) as output_file:
#         print(f"...Writing to CSV...")
#         dict_writer = csv.DictWriter(output_file, keys)
#         dict_writer.writeheader()
#         dict_writer.writerows(dict)
#     await session.close()
