from dataModel.player import Player
from utils import fetch
from constants import endpoints

STATIC_BASE_URL = endpoints["STATIC"]["BASE_URL"]


class Team:
    def __init__(self, team_information, session):
        self.session = session
        for k, v in team_information.items():
            setattr(self, k, v)

    async def get_players_for_team(self, return_json=False):
        """Returns a list containing the players who play for the team. Does
        not include the player's summary.

        :param return_json: (optional) Boolean. If ``True`` returns a list of
            dicts, if ``False`` returns a list of Player objects. Defaults to
            ``False``.
        :type return_json: bool
        :rtype: list
        """
        team_players = getattr(self, "players", [])

        if not team_players:
            players = await fetch(self.session, STATIC_BASE_URL)
            players = players["elements"]
            team_players = [player for player in players if player["team"] == self.id]
            self.players = team_players

        if return_json:
            return team_players

        return [Player(player) for player in team_players]
