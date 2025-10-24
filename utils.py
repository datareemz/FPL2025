from asyncio import exceptions
from constants import endpoints
from json import JSONDecodeError
import requests
import certifi
import ssl
import http
from http import cookies
import secrets
import hashlib
import base64
from google.cloud import secretmanager
import json
import os


headers = {
    "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 5.1; PRO 5 Build/LMY47D)",
    "accept-language": "en",
}

cookies = http.cookies.SimpleCookie()
ssl_context = ssl.create_default_context(cafile=certifi.where())


STATIC_BASE_URL = endpoints["STATIC"]["BASE_URL"]


def headers_access(access_token):
    headers["Authorization"] = "Bearer {}".format(access_token)
    return headers


async def fetch(session, url, headers=None):
    while True:
        try:
            async with session.get(
                url, headers=headers, ssl=ssl_context, cookies=cookies
            ) as response:
                assert response.status == 200
                return await response.json(content_type=None)
        except Exception as e:
            raise e


async def post(session, url, payload, headers):
    async with session.post(url, data=payload, headers=headers) as response:
        return await response.json()


def position_converter(position):
    """Converts a player's `element_type` to their actual position."""
    position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    return position_map[position]


def team_converter(team_id):
    try:
        dynamic = requests.get(STATIC_BASE_URL).json()
    except exceptions.RequestException as e:
        raise e
    teamname_list = [team["name"] for team in dynamic["teams"]]
    teamdict = {
        dynamic["teams"][i]["id"]: dynamic["teams"][i]["name"]
        for i in range(len(teamname_list))
    }
    return teamdict[team_id]


def convertTeamForm(form: str):
    res = 0
    for i in form:
        if i == "W":
            res += 3
        if i == "D":
            res += 1
        if i == "L":
            res += 0
    return float(res / (len(form) * 3))


def get_teams():
    dynamic = requests.get(STATIC_BASE_URL).json()
    teamname_list = [team["name"] for team in dynamic["teams"]]
    return {
        dynamic["teams"][i]["id"]: dynamic["teams"][i]["name"]
        for i in range(len(teamname_list))
    }


def get_team(team_id, team_dict=None):
    if team_dict:
        return team_dict.get(team_id)


def get_headers(referer):
    """Returns the headers needed for the transfer request."""
    return {
        "Content-Type": "application/json;charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": referer,
    }


# def get_transfer_candidates(my_team, player_pool,max=3):
async def post(session, url, payload, headers):
    async with session.post(url, data=payload, headers=headers) as response:
        return await response.json()


async def post_transfer(session, url, payload, headers):
    async with session.post(url, data=payload, headers=headers) as response:
        if response.status == 200:
            return
        try:
            result = await response.json(content_type=None)
        except JSONDecodeError:
            result = await response.text()
            raise Exception(
                f"Unknown error while requesting {response.url}. {response.status} - {result}"
            )

        if result.get("errorCode"):
            message = result.get("error")

            raise Exception(message if message else result)


def generate_code_verifier():
    return secrets.token_urlsafe(64)[:128]


def generate_code_challenge(verifier):
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def get_credentials_from_secret_manager():
    """Fetch credentials from Google Secret Manager.

    Returns:
        tuple: (email, password) retrieved from Secret Manager
    """
    try:
        project_id = os.getenv("GCP_PROJECT_ID", "ardent-quarter-468720-i2")
        secret_id = os.getenv("SECRET_ID", "fpl_2025_credentials")

        client = secretmanager.SecretManagerServiceClient()
        secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

        response = client.access_secret_version(request={"name": secret_name})
        secret_data = response.payload.data.decode("UTF-8")

        credentials = json.loads(secret_data)
        return credentials.get("email"), credentials.get("password")
    except Exception as e:
        print(f"Error fetching credentials from Secret Manager: {e}")
        return None, None
