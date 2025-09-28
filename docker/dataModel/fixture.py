from utils import get_team


class Fixture:
    """A class representing a fixture in Fantasy Premier League."""

    def __init__(self, fixture_information, team_dict):
        for k, v in fixture_information.items():
            if k == "team_a":
                v = get_team(team_dict=team_dict, team_id=v)
            if k == "team_h":
                v = get_team(team_dict=team_dict, team_id=v)
            if k == "stats" and isinstance(
                v, list
            ):  # Historical Does not Have Stats in list format
                v = {w["identifier"]: {"a": w["a"], "h": w["h"]} for w in v}

            setattr(self, k, v)

    def get_home_team(self):
        return getattr(self, "team_h")

    def get_away_team(self):
        return getattr(self, "team_a")

    def is_draw(self):
        try:
            assert getattr(self, "team_a_score") == getattr(self, "team_h_score")
            return True
        except AssertionError:
            return False

    def get_winner(self):
        try:
            assert not self.is_draw()
            return (
                self.get_away_team()
                if getattr(self, "team_a_score") > getattr(self, "team_h_score")
                else self.get_home_team()
            )

        except AssertionError:
            return False

    def __str__(self):
        return f"{self.team_h} vs. " f"{self.team_a} - " f"{self.kickoff_time}"
