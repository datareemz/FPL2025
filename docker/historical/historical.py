from dataModel.fixture import Fixture
import csv
import utils

headers = utils.headers

final = []


def getHistoricalTeamDict(season):
    teamdict = {}
    with open(
        "../historical/_teams/teams_{}.csv".format(season), newline=""
    ) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            teamdict[row["id"]] = row["name"]
    return teamdict


def getHistoricalFixtures(season, team_dict):
    hist_fixtures = []
    with open(
        "../historical/_fixtures/fixtures_{}.csv".format(season), newline=""
    ) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            hist_fixtures.append(row)
        return [Fixture(fixture, team_dict=team_dict) for fixture in hist_fixtures]


def getFormDict(season=None, fixtures=None):
    if season:
        team_dict = getHistoricalTeamDict(season)
        fixtures = getHistoricalFixtures(season, team_dict)

    else:
        team_dict = utils.get_teams()
        fixtures = fixtures

    form_dict = {}
    score_strength_dict = {}

    for i in team_dict:
        form_dict[team_dict[i]] = ""
        score_strength_dict[team_dict[i]] = 0
        for j in fixtures:
            if team_dict[i] == j.get_away_team() or team_dict[i] == j.get_home_team():
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


def get_FDR(form_dict, fixtures=None, season=None):
    if season:
        team_dict = getHistoricalTeamDict(season)
        fixtures = getHistoricalFixtures(season, team_dict)
    else:
        fixtures = fixtures
    fdr_dict = {}
    for i in form_dict:
        fdr_dict[i] = 0
        for j in fixtures:
            if i == j.get_away_team():
                fdr_dict[i] += utils.convertTeamForm(
                    form_dict[i]
                ) - utils.convertTeamForm(form_dict[j.get_home_team()])
            if i == j.get_home_team():
                fdr_dict[i] += utils.convertTeamForm(
                    form_dict[i]
                ) - utils.convertTeamForm(form_dict[j.get_away_team()])

    return fdr_dict
