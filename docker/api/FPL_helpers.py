import asyncio
import warnings
import itertools
import csv
import pandas as pd
import logging
import utils
import os
from constants import (
    endpoints,
    PLAYER_DATA_SCHEMA,
    POSITION_WEIGHTS,
    MAX_DEF,
    MAX_GK,
    MAX_MID,
    MAX_FWD,
    MAX_BUDGET,
    MAX_PLAYER_FROM_TEAM,
    FPL_BUCKET,
)
from dataModel.player import Player
from dataModel.team import Team
from dataModel.fixture import Fixture

# Get logger (configuration done in main entry point)
logger = logging.getLogger(__name__)

STATIC_BASE_URL = endpoints["STATIC"]["BASE_URL"]
API_GW_FIXTURES = endpoints["API"]["GW_FIXTURES"]
API_ALL_FIXTURES = endpoints["API"]["ALL_FIXTURES"]

is_c = "is_captain"
is_vc = "is_vice_captain"


class FPLHelpers:
    """Helper class for FPL operations that don't require authentication."""

    def __init__(self, session):
        self.session = session

    async def get_all_current_players(self, player_ids=None, return_json=False):
        """Returns either a list of *all* players, or a list of players whose
        IDs are in the given ``player_ids`` list.

        Information is taken from e.g.:
            https://fantasy.premierleague.com/api/bootstrap-static/
            https://fantasy.premierleague.com/api/element-summary/1/ (optional)

        :param list player_ids: (optional) A list of player IDs
        :param boolean include_summary: (optional) Includes a player's summary
            if ``True``.
        :param return_json: (optional) Boolean. If ``True`` returns a list of
            ``dict``s, if ``False`` returns a list of  :class:`Player`
            objects. Defaults to ``False``.
        :type return_json: bool
        :rtype: list
        """
        try:
            data = await utils.fetch(self.session, STATIC_BASE_URL)
            players = data["elements"]
        except Exception as e:
            logger.error("Error fetching players data: %s", e)
            raise
        if player_ids:
            players = [player for player in players if player["id"] in player_ids]
            if not return_json:
                return players
            else:
                return [Player(player) for player in players]
        if not player_ids:
            tasks = [
                asyncio.ensure_future(self.get_current_player(player=player))
                for player in players
            ]
            players = await asyncio.gather(*tasks)
            return players

    async def get_current_player(
        self, player_id=None, player=None, return_json=False, convert_hist=False
    ):
        """Returns the player with the given ``player_id``.

        :param player:
        :param return_json:
        :param convert_hist:
        :param player_id: A player's ID.
        :type player_id: string or int
        :rtype: :class:`Player` or ``dict``
        :raises ValueError: Player with ``player_id`` not found
        """
        data = await utils.fetch(self.session, STATIC_BASE_URL)
        players = data["elements"]
        if not player and player_id:
            try:
                player = next(player for player in players if player["id"] == player_id)
            except StopIteration:
                raise ValueError(f"Player with ID {player_id} not found")
            if return_json:
                return player
        if player and convert_hist:
            try:
                player = next(
                    x
                    for x in players
                    if (
                        x["first_name"] == player.first_name
                        and x["second_name"] == player.second_name
                    )
                    or x["web_name"] == player.first_name + " " + player.second_name
                )
            except Exception as e:
                warnings.warn(
                    f"Player name {player.first_name} {player.second_name} not found, Might not be in the EPL"
                    f"anymore Dawg"
                )
                logger.error("Error finding player: %s", e)
        return Player(player)

    async def get_team(self, *team_ids, team_names=None):
        try:
            response = await utils.fetch(self.session, STATIC_BASE_URL)
        except Exception:
            raise Exception("Failed to fetch team data")
        teams = response["teams"]
        if team_ids:
            team = (team for team in teams if team["id"] in team_ids)
        else:
            team = (team for team in teams if team["name"] in team_names)
        return [Team(team, self.session) for team in team]

    async def prepare_data(self, historical=False):
        data_dict = []
        try:
            response = await utils.fetch(self.session, STATIC_BASE_URL)
            team_dict = utils.get_teams()
        except Exception:
            raise Exception("Failed to fetch team data")
        sorted_players = response["elements"]
        for baller in sorted_players:
            goals_scored = baller["goals_scored"]
            assists = baller["assists"]
            goal_contributions = goals_scored + assists
            starts = baller["starts"]
            minutes = baller["minutes"]

            teamname = utils.get_team(baller["team"], team_dict)
            roi = float(f"{(baller['total_points'] / (baller['now_cost'] / 10)):000.4}")
            roi_per_gw = roi / starts if starts > 0 else 0
            # pos = utils.position_converter(baller["element_type"])
            diction = PLAYER_DATA_SCHEMA.copy()
            diction.update(
                {
                    "first_name": baller["first_name"],
                    "second_name": baller["second_name"],
                    "now_cost": round(baller["now_cost"] / 10, 2),
                    "starts": int(starts),
                    "goals_scored": int(goals_scored),
                    "assists": int(assists),
                    "goal_contributions": int(goal_contributions),
                    "total_points": int(baller["total_points"]),
                    "points_per_game": float(baller["points_per_game"]),
                    "roi": float(roi),
                    "roi_per_gw": float(roi_per_gw),
                    "element_type": baller["element_type"],
                    "team": baller["team"],
                    "team_name": teamname,
                    "minutes": int(minutes),
                    "FDR_Average": 0.0,
                    "id": baller["id"],
                }
            )

            data_dict.append(diction)
        return data_dict

    @staticmethod
    def set_captain(lineup, captain, captain_type, player_ids):
        """Sets the given captain's captain_type to True.

        :param lineup: List of players.
        :type lineup: list
        :param captain: ID of the captain.
        :type captain: int or str
        :param captain_type: The captain type: 'is_captain' or 'is_vice_captain'.
        :type captain_type: string
        :param player_ids: List of the team's players' IDs.
        :type player_ids: list
        """
        if captain and captain not in player_ids:
            raise ValueError("Cannot (vice) captain player who isn't in user's team.")

        current_captain = next(player for player in lineup if player[captain_type])
        chosen_captain = next(
            player for player in lineup if player["element"] == captain
        )

        # If the chosen captain is already a (vice) captain, then give his previous
        # role to the current (vice) captain.
        if chosen_captain[is_c] or chosen_captain[is_vc]:
            current_captain[is_c], chosen_captain[is_c] = (
                chosen_captain[is_c],
                current_captain[is_c],
            )
            current_captain[is_vc], chosen_captain[is_vc] = (
                chosen_captain[is_vc],
                current_captain[is_vc],
            )

        for player in lineup:
            player[captain_type] = False

            if player["element"] == captain:
                player[captain_type] = True

    async def get_data(self, game_week):
        file_name = f"datastore/current/FPL_data_{game_week}.csv"
        if not utils.check_file_exists_google_cloud(
            bucket_name=FPL_BUCKET, file_name=file_name
        ):
            fixtures = await self.get_all_fixtures(*range(1, game_week))
            f, s = self.get_form_dict(fixtures=fixtures)
            logger.info("form_dict: %s", f)
            logger.info("score_strength(not used): %s", s)
            g = self.get_FDR(form_dict=f, fixtures=fixtures)
            logger.info("FDR dictionary: %s", g)
            player_data = await self.prepare_data()
            for entry in player_data:
                if game_week == 1:
                    entry["starts"] = 0
                    entry["minutes"] = 0
                if entry["team_name"] in g:
                    entry["FDR_Average"] = round(g[entry["team_name"]], 3)
            keys = player_data[0].keys()
            with open(file_name, "w", newline="") as output_file:
                logger.info("Writing to CSV...")
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(player_data)
            utils.write_file_to_google_storage(
                bucket_name=FPL_BUCKET,
                source_file_name=file_name,
                destination_blob_name=file_name,
            )
        else:
            utils.read_file_from_google_storage(
                bucket_name=FPL_BUCKET,
                source_blob_name=file_name,
                destination_file_name=file_name,
            )

    async def get_all_fixtures(self, *game_week):
        """Returns all fixtures for the specified game weeks.

        :param game_week: Game week numbers to fetch fixtures for
        :rtype: list
        """
        task = asyncio.ensure_future(utils.fetch(self.session, API_ALL_FIXTURES))

        gameweek_fixtures = await asyncio.gather(task)
        fixtures = list(itertools.chain(*gameweek_fixtures))

        team_dict = utils.get_teams()
        return [
            Fixture(fixture, team_dict)
            for fixture in fixtures
            if fixture["event"] in game_week
        ]

    async def get_upcoming_gameweek(self):
        """Returns the upcoming game week number.

        :rtype: int
        """
        try:
            response = await utils.fetch(self.session, API_GW_FIXTURES)
            gw = max([x["event"] for x in response if x["finished"] is True])
            # This is the previous game week
        except Exception:
            warnings.warn("Start of the season, game week 1")
            return 1
        if len(response) == 0:
            raise Exception("No Active Events yet.... TODO")
        return int(gw) + 1  # Adds one to previous game week

    async def get_game_week_stats(self, gw):
        """Returns the stats for game week.
        :rtype: dictionary
        """
        stats = ["id", "average_entry_score", "highest_score"]
        try:
            response = await utils.fetch(self.session, STATIC_BASE_URL)
            gw_stats = [x for x in response["events"] if x["can_manage"] is False]
            gw_stats = next(x for x in gw_stats if x["id"] == gw)
            gw_stats = {key: value for key, value in gw_stats.items() if key in stats}
        except Exception:
            warnings.warn("Start of the season, game week 1")
            return 1
        return gw_stats

    def get_historical_team_dict(self, season):
        with open(f"../historical/_teams/teams_{season}.csv", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            return {row["id"]: row["name"] for row in reader}

    def get_historical_fixtures(self, season, team_dict):
        with open(
            f"../historical/_fixtures/fixtures_{season}.csv", newline=""
        ) as csvfile:
            reader = csv.DictReader(csvfile)
            return [Fixture(row, team_dict=team_dict) for row in reader]

    def get_form_dict(self, season=None, fixtures=None):
        if season:
            team_dict = self.get_historical_team_dict(season)
            fixtures = self.get_historical_fixtures(season, team_dict)

        else:
            team_dict = utils.get_teams()
            fixtures = fixtures

        form_dict = {}
        score_strength_dict = {}

        for i in team_dict:
            form_dict[team_dict[i]] = ""
            score_strength_dict[team_dict[i]] = 0
            for j in fixtures:
                if (
                    team_dict[i] == j.get_away_team()
                    or team_dict[i] == j.get_home_team()
                ):
                    if j.get_winner() == team_dict[i]:
                        form_dict[team_dict[i]] += "W"
                        if abs(int(j.team_h_score) - int(j.team_a_score)) >= 3:
                            score_strength_dict[team_dict[i]] += 1
                    if j.is_draw():
                        form_dict[team_dict[i]] += "D"
                    if not j.is_draw() and j.get_winner() != team_dict[i]:
                        form_dict[team_dict[i]] += "L"
                        if abs(int(j.team_h_score) - int(j.team_a_score)) >= 3:
                            score_strength_dict[team_dict[i]] -= 1
        return form_dict, score_strength_dict

    def get_FDR(self, form_dict, fixtures=None, season=None):
        if season:
            team_dict = self.get_historical_team_dict(season)
            fixtures = self.get_historical_fixtures(season, team_dict)
        else:
            fixtures = fixtures
        fdr_dict = {}
        for i in form_dict:
            fdr_dict[i] = 0
            for j in fixtures:
                if i == j.get_away_team():
                    fdr_dict[i] += utils.convert_team_form(
                        form_dict[i]
                    ) - utils.convert_team_form(form_dict[j.get_home_team()])
                if i == j.get_home_team():
                    fdr_dict[i] += utils.convert_team_form(
                        form_dict[i]
                    ) - utils.convert_team_form(form_dict[j.get_away_team()])

        return fdr_dict

    async def get_fixtures_for_gameweek(self, game_week: int):
        """Returns the fixtures for the current game week.

        :param game_week: The game week number
        :type game_week: int
        :rtype: list
        """
        try:
            response = await utils.fetch(
                self.session, API_GW_FIXTURES.format(f=game_week)
            )
        except Exception:
            raise Exception("Failed to fetch fixtures for game week")

        team_dict = utils.get_teams()
        fixtures = [x for x in response if x["event"] == game_week]
        return [Fixture(fixture, team_dict=team_dict) for fixture in fixtures]

    # def get_optimal_transfers(self, current_team, player_pool, max_transfers=3, metric="total_points", min_improvement=0.1):
    #     """Get optimal transfers for the current team from available player pool.
    #
    #     :param current_team: List of current Player objects
    #     :param player_pool: List of available Player objects
    #     :param max_transfers: Maximum number of transfers to suggest
    #     :param metric: Optimization metric ('total_points', 'roi', 'points_per_Min')
    #     :param min_improvement: Minimum improvement required to suggest transfer
    #     :return: Dict with transfer recommendations
    #     """
    #     return optimize_team_transfers(current_team, player_pool, max_transfers, metric, min_improvement)
    #
    # async def get_optimal_transfers_with_fixtures(self, current_team, player_pool, upcoming_gameweeks=5, max_transfers=3, metric="projected_points", min_improvement=0.1):
    #     """Get optimal transfers considering upcoming fixtures and predicted points.
    #
    #     :param current_team: List of current Player objects
    #     :param player_pool: List of available Player objects
    #     :param upcoming_gameweeks: Number of gameweeks to consider for projections
    #     :param max_transfers: Maximum number of transfers to suggest
    #     :param metric: Optimization metric ('projected_points', 'form_adjusted', 'fixture_adjusted')
    #     :param min_improvement: Minimum improvement required to suggest transfer
    #     :return: Dict with transfer recommendations including fixture analysis
    #     """
    #     try:
    #         current_gw = await self.get_upcoming_gameweek()
    #         fixtures = await self.get_all_fixtures(*range(current_gw, current_gw + upcoming_gameweeks))
    #
    #         return optimize_team_transfers_with_fixtures(
    #             current_team, player_pool, fixtures, upcoming_gameweeks,
    #             max_transfers, metric, min_improvement
    #         )
    #     except Exception as e:
    #         # Fallback to basic optimization if fixture data unavailable
    #         return optimize_team_transfers(current_team, player_pool, max_transfers, "total_points", min_improvement)

    def validate_team_constraints(self, team=None):
        """Validates that a team meets all FPL constraints.

        :param team: List of Player objects
        :return: Dict with constraint validation results
        """
        if team is None:
            team = [Player]
        if not team or len(team) != 15:
            return {
                "valid": False,
                "error": f"Team must have exactly 15 players, has {len(team)}",
            }

        # Count positions
        position_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}  # GK, DEF, MID, FWD
        team_counts = {}
        total_cost = 0

        for player in team:

            pos = player._position()
            position_counts[pos] = position_counts.get(pos, 0) + 1
            team_counts[player.team] = team_counts.get(player.team, 0) + 1
            total_cost += player.now_cost

        # Check position constraints
        if position_counts.get("GK", 0) != MAX_GK:
            return {
                "valid": False,
                "error": f"Must have {MAX_GK} goalkeepers, has {position_counts.get('GK', 0)}",
            }
        if position_counts.get("DEF", 0) != MAX_DEF:
            return {
                "valid": False,
                "error": f"Must have {MAX_DEF} defenders, has {position_counts.get('DEF', 0)}",
            }
        if position_counts.get("MID", 0) != MAX_MID:
            return {
                "valid": False,
                "error": f"Must have {MAX_MID} midfielders, has {position_counts.get('MID', 0)}",
            }
        if position_counts.get("FWD", 0) != MAX_FWD:
            return {
                "valid": False,
                "error": f"Must have {MAX_FWD} forwards, has {position_counts.get('FWD', 0)}",
            }

        # Check budget constraint
        if total_cost > MAX_BUDGET:
            return {
                "valid": False,
                "error": f"Team cost {total_cost:.1f} exceeds budget {MAX_BUDGET}",
            }

        # Check team constraint
        for team_id, count in team_counts.items():
            if count > MAX_PLAYER_FROM_TEAM:
                return {
                    "valid": False,
                    "error": f"Team {team_id} has {count} players, max is {MAX_PLAYER_FROM_TEAM}",
                }

        return {
            "valid": True,
            "position_counts": position_counts,
            "team_counts": team_counts,
            "total_cost": total_cost,
            "remaining_budget": MAX_BUDGET - total_cost,
        }

    def _get_metric_value(self, player, metric):
        """Extracts a metric value, handling both attributes and methods.

        :param player: Player object
        :param metric: Metric name string
        :return: float metric value
        """
        if player is None:
            return 0.0
        if not isinstance(metric, str):
            return 0.0
        val = getattr(player, metric, 0)
        if callable(val):
            return float(val())
        return float(val)

    # O(n * m) - n players, m metrics
    def get_team_analysis(self, team, metrics):
        """Analyzes team using z-score normalization and position weights.

        :param team: List of Player objects
        :param metrics: List of metric names
        :return: Dict with team analysis and metric_stats for reuse
        """
        if not team:
            return {"by_position": {}, "no_position": [], "metric_stats": {}}
        if not metrics:
            return {"by_position": {}, "no_position": [], "metric_stats": {}}

        analysis = {
            "by_position": {1: [], 2: [], 3: [], 4: []},
            "weakest_players": [],
            "no_position": [],
            "metrics_used": metrics,
            "metric_stats": {},
        }

        # Collect raw metric values per player
        raw_values = {m: [] for m in metrics}
        player_raw = []
        for player in team:
            raw = {}
            for metric in metrics:
                val = self._get_metric_value(player, metric)
                raw[metric] = val
                raw_values[metric].append(val)
            player_raw.append((player, raw))

        # Compute team-wide mean and std per metric
        metric_stats = {}
        for metric in metrics:
            values = raw_values[metric]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5
            metric_stats[metric] = {"mean": mean, "std": std}
        analysis["metric_stats"] = metric_stats

        # Z-score normalize and apply position-specific weights
        all_players = []
        for player, raw in player_raw:
            position = getattr(player, "element_type", 3)
            weights = POSITION_WEIGHTS.get(position, POSITION_WEIGHTS[3])

            total_value = 0.0
            z_scores = {}
            for metric in metrics:
                stats = metric_stats[metric]
                z = 0.0
                if stats["std"] > 0:
                    z = (raw[metric] - stats["mean"]) / stats["std"]
                z_scores[metric] = z
                total_value += z * weights.get(metric, 0.0)

            player_info = {
                "player_object": player,
                "player_name": str(player),
                "total_metric_value": total_value,
                "individual_metrics": z_scores,
                "raw_metrics": raw,
            }
            analysis["by_position"][position].append(player_info)
            all_players.append(player_info)

        # Sort by metric within each position, weakest first
        for position in analysis["by_position"]:
            analysis["by_position"][position].sort(
                key=lambda x: x["total_metric_value"]
            )
            if analysis["by_position"][position]:
                analysis["weakest_players"].append(
                    analysis["by_position"][position][0]
                )

        analysis["weakest_players"].sort(key=lambda x: x["total_metric_value"])
        all_players.sort(key=lambda x: x["total_metric_value"])
        analysis["no_position"] = all_players

        return analysis


    def find_valid_replacement(
        self, player_out, player_pool, current_team, metrics, metric_stats
    ):
        """Finds valid replacement using absolute z-score comparison.

        :param player_out: Dict from get_team_analysis with player_object and total_metric_value
        :param player_pool: Available Player objects
        :param current_team: Current team list
        :param metrics: List of metric names
        :param metric_stats: Dict of {metric: {mean, std}} from get_team_analysis
        :return: Sorted list of candidate dicts or None
        """
        if not player_pool or not metric_stats:
            return None
        if not player_out:
            return None

        player_out_obj = player_out["player_object"]
        player_out_score = player_out["total_metric_value"]
        position = getattr(player_out_obj, "element_type", 3)
        weights = POSITION_WEIGHTS.get(position, POSITION_WEIGHTS[3])

        # Filter pool to same position, exclude current team
        candidates = [
            p for p in player_pool
            if p._position() == player_out_obj._position()
        ]
        _ids_in_team = {
            getattr(p, "id", None) for p in current_team
        }
        candidates = [
            p for p in candidates
            if getattr(p, "id", None) not in _ids_in_team
        ]

        valid_candidates = []
        for candidate in candidates:
            temp_team = [
                p for p in current_team if p != player_out_obj
            ] + [candidate]
            team_validation = self.validate_team_constraints(temp_team)

            if not team_validation["valid"]:
                continue

            # Score candidate on the same z-score scale
            candidate_score = 0.0
            individual_metrics = {}
            for metric in metrics:
                val = self._get_metric_value(candidate, metric)
                stats = metric_stats[metric]
                z = 0.0
                if stats["std"] > 0:
                    z = (val - stats["mean"]) / stats["std"]
                individual_metrics[metric] = z
                candidate_score += z * weights.get(metric, 0.0)

            cost_diff = candidate.now_cost - player_out_obj.now_cost
            improvement = candidate_score - player_out_score

            valid_candidates.append({
                "player": candidate,
                "cost_diff": cost_diff,
                "individual_metrics": individual_metrics,
                "candidate_score": candidate_score,
                "total_metric_improvement_val": improvement,
            })

        if valid_candidates:
            valid_candidates.sort(
                key=lambda x: x["total_metric_improvement_val"],
                reverse=True,
            )
            return valid_candidates

        return None

    # def optimize_team_transfers(self, current_team, player_pool, max_transfers=3, metric="total_points", min_improvement=0.1):
    #     """Optimizes team by suggesting transfers that improve performance while maintaining constraints.
    #
    #     :param current_team: List of current Player objects
    #     :param player_pool: List of available Player objects
    #     :param max_transfers: Maximum number of transfers to suggest
    #     :param metric: Optimization metric ('total_points', 'roi', 'points_per_Min')
    #     :param min_improvement: Minimum improvement required to suggest transfer
    #     :return: Dict with transfer recommendations
    #     """
    #     # Validate current team
    #     team_validation = self.validate_team_constraints(current_team)
    #     if not team_validation["valid"]:
    #         return {"error": f"Current team invalid: {team_validation['error']}"}
    #
    #     # Analyze current team
    #     team_analysis = self.get_team_analysis(current_team, [metric])
    #
    #     suggested_transfers = []
    #     temp_team = current_team.copy()
    #
    #     # Try to make transfers for weakest players
    #     for weak_player_info in team_analysis["weakest_players"]:
    #         if len(suggested_transfers) >= max_transfers:
    #             break
    #
    #         player_out = weak_player_info["player"]
    #
    #         # Find best replacement
    #         replacement = self.find_valid_replacement(player_out, player_pool, temp_team, metric)
    #
    #         if replacement:
    #             # Calculate improvement
    #             if metric == "total_points":
    #                 current_value = getattr(player_out, "total_points", 0)
    #                 new_value = getattr(replacement, "total_points", 0)
    #             elif metric == "roi":
    #                 current_value = player_out.roi()
    #                 new_value = replacement.roi()
    #             elif metric == "points_per_Min":
    #                 current_value = player_out.points_per_Min()
    #                 new_value = replacement.points_per_Min()
    #             else:
    #                 current_value = getattr(player_out, metric, 0)
    #                 new_value = getattr(replacement, metric, 0)
    #
    #             improvement = new_value - current_value
    #
    #             if improvement >= min_improvement:
    #                 suggested_transfers.append({
    #                     "player_out": player_out,
    #                     "player_in": replacement,
    #                     "improvement": improvement,
    #                     "cost_change": replacement.now_cost - player_out.now_cost,
    #                     "position": position_converter(player_out.element_type)
    #                 })
    #
    #                 # Update temp team for next iteration
    #                 temp_team = [p for p in temp_team if p != player_out] + [replacement]
    #
    #     # Calculate total improvement
    #     total_improvement = sum(t["improvement"] for t in suggested_transfers)
    #     total_cost_change = sum(t["cost_change"] for t in suggested_transfers)
    #
    #     return {
    #         "transfers": suggested_transfers,
    #         "total_improvement": total_improvement,
    #         "total_cost_change": total_cost_change,
    #         "new_budget_used": team_validation["total_cost"] + total_cost_change,
    #         "metric_used": metric,
    #         "current_team_analysis": team_analysis
    #     }

    # def calculate_fdr_for_upcoming_fixtures(self, team_name, fixtures, gameweeks=5):
    #     """Calculate FDR-based fixture difficulty for a team using your existing FDR system.
    #
    #     :param team_name: Team name to analyze
    #     :param fixtures: List of Fixture objects
    #     :param gameweeks: Number of gameweeks to consider
    #     :return: Dict with FDR-based difficulty metrics
    #     """
    #     # Get current form and FDR data
    #     try:
    #         form_dict, score_strength_dict = get_form_dict(fixtures=fixtures)
    #         fdr_dict = get_FDR(form_dict=form_dict, fixtures=fixtures)
    #     except Exception:
    #         # Fallback if FDR calculation fails
    #         return {
    #             'team_fdr': 0.0,
    #             'home_fixtures': 0,
    #             'away_fixtures': 0,
    #             'total_fixtures': 0,
    #             'fdr_score': 0.0,
    #             'fixtures': []
    #         }
    #
    #     # Get team's overall FDR
    #     team_fdr = fdr_dict.get(team_name, 0.0)
    #
    #     # Analyze upcoming fixtures specifically
    #     team_fixtures = []
    #     for fixture in fixtures:
    #         home_team = getattr(fixture, 'team_h', '') or fixture.get_home_team()
    #         away_team = getattr(fixture, 'team_a', '') or fixture.get_away_team()
    #
    #         if team_name == home_team or team_name == away_team:
    #             is_home = (team_name == home_team)
    #             opponent = away_team if is_home else home_team
    #
    #             # Calculate specific fixture FDR (team form - opponent form)
    #             team_form = form_dict.get(team_name, "")
    #             opponent_form = form_dict.get(opponent, "")
    #
    #             from historical.historical import convertTeamForm
    #             team_strength = convertTeamForm(team_form)
    #             opponent_strength = convertTeamForm(opponent_form)
    #
    #             fixture_fdr = team_strength - opponent_strength
    #             # Add small home advantage bonus
    #             if is_home:
    #                 fixture_fdr += 0.1
    #
    #             team_fixtures.append({
    #                 'gameweek': getattr(fixture, 'event', 0),
    #                 'is_home': is_home,
    #                 'opponent': opponent,
    #                 'fixture_fdr': fixture_fdr,
    #                 'team_form': team_form,
    #                 'opponent_form': opponent_form
    #             })
    #
    #     # Sort by gameweek and take only upcoming fixtures
    #     team_fixtures.sort(key=lambda x: x['gameweek'])
    #     upcoming_fixtures = team_fixtures[:gameweeks]
    #
    #     if not upcoming_fixtures:
    #         return {
    #             'team_fdr': team_fdr,
    #             'home_fixtures': 0,
    #             'away_fixtures': 0,
    #             'total_fixtures': 0,
    #             'fdr_score': team_fdr,
    #             'fixtures': []
    #         }
    #
    #     # Calculate metrics for upcoming fixtures
    #     avg_fixture_fdr = sum(f['fixture_fdr'] for f in upcoming_fixtures) / len(upcoming_fixtures)
    #     home_fixtures = sum(1 for f in upcoming_fixtures if f['is_home'])
    #     away_fixtures = len(upcoming_fixtures) - home_fixtures
    #
    #     return {
    #         'team_fdr': team_fdr,  # Overall team FDR
    #         'avg_fixture_fdr': avg_fixture_fdr,  # Average FDR for upcoming fixtures
    #         'home_fixtures': home_fixtures,
    #         'away_fixtures': away_fixtures,
    #         'total_fixtures': len(upcoming_fixtures),
    #         'fdr_score': avg_fixture_fdr,  # Use upcoming fixture FDR as main score
    #         'fixtures': upcoming_fixtures
    #     }
