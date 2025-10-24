import os
import uuid
import re
import json
import utils
from constants import endpoints
from dataModel.user import User
from google.cloud import secretmanager

LOGIN_URL = endpoints["API"]["LOGIN"]
AUTH_URL = endpoints["API"]["AUTH"]
API_ME = endpoints["API"]["ME"]
REDIRECT_URI = endpoints["API"]["REDIRECT_URI"]
DAVINCI_POLICY_START_URL = endpoints["API"]["DAVINCI_POLICY_START"]
DAVINCI_CONNECTIONS_URL = endpoints["API"]["DAVINCI_CONNECTIONS"]
AS_RESUME_URL = endpoints["API"]["AS_RESUME"]
AS_TOKEN_URL = endpoints["API"]["AS_TOKEN"]
STANDARD_CONNECTION_ID = "0d8c928e4970386733ce110b9dda8412"


class FPLAuth:
    """Handles FPL authentication and access token"""

    def __init__(self, session):
        self.session = session
        self.user = None

    def _get_credentials_from_secret_manager(self):
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

    async def login(self, email=None, password=None):
        """Returns a requests session with FPL login authentication.

        :param string email: Email address for the user's Fantasy Premier League
            account.
        :param string password: Password for the user's Fantasy Premier League
            account.
        """
        if not email and not password:
            email, password = self._get_credentials_from_secret_manager()
            if not email or not password:
                email = os.getenv("FPL_EMAIL")
                password = os.getenv("FPL_PASSWORD")
        if not email or not password:
            raise ValueError(
                "Email and password must be set. Configure them in Google Secret Manager or environment variables."
            )
        print(f"Logging in: {LOGIN_URL}")

        code_verifier = utils.generate_code_verifier()
        code_challenge = utils.generate_code_challenge(code_verifier)

        initial_state = uuid.uuid4().hex

        params = {
            "response_type": "code",
            "scope": "openid profile email offline_access",
            "client_id": "bfcbaf69-aade-4c1b-8f00-c1cb8a193030",
            "redirect_uri": REDIRECT_URI,
            "state": initial_state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        async with self.session.get(AUTH_URL, params=params) as response:
            assert response.status == 200
            if response.status == 200:
                text2 = await response.text()
                access_token = re.search(r'"accessToken":"([^"]+)"', text2).group(1)
                new_state = re.search(
                    r'<input[^>]+name="state"[^>]+value="([^"]+)"', text2
                ).group(1)
                print("Successfully Retrieved Access Token... Continuing...")
            else:
                if response.status != 200:
                    raise Exception("Incorrect email or password!")

        new_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        async with self.session.post(
            DAVINCI_POLICY_START_URL, headers=new_headers
        ) as response:
            assert response.status == 200
            response = await response.json()
            interaction_id = response["interactionId"]
            # interactionToken = response["interactionToken"]

        async with self.session.post(
            DAVINCI_CONNECTIONS_URL.format(STANDARD_CONNECTION_ID),
            headers={
                "interactionId": interaction_id,
                # "interactionToken": interaction_token,
            },
            json={
                "id": response["id"],
                "eventName": "continue",
                "parameters": {"eventType": "polling"},
                "pollProps": {
                    "status": "continue",
                    "delayInMs": 10,
                    "retriesAllowed": 1,
                    "pollChallengeStatus": False,
                },
            },
        ) as response:
            assert response.status == 200
            response = await response.json()

        async with self.session.post(
            DAVINCI_CONNECTIONS_URL.format(STANDARD_CONNECTION_ID),
            headers={
                "interactionId": interaction_id,
                # "interactionToken": interaction_token,
            },
            json={
                "id": response["id"],
                "nextEvent": {
                    "constructType": "skEvent",
                    "eventName": "continue",
                    "params": [],
                    "eventType": "post",
                    "postProcess": {},
                },
                "parameters": {
                    "buttonType": "form-submit",
                    "buttonValue": "SIGNON",
                    "username": email,
                    "password": password,
                },
                "eventName": "continue",
            },
        ) as response:
            assert response.status == 200
            response = await response.json()

        async with self.session.post(
            DAVINCI_CONNECTIONS_URL.format(response["connectionId"]),
            headers=new_headers,
            json={
                "id": response["id"],
                "nextEvent": {
                    "constructType": "skEvent",
                    "eventName": "continue",
                    "params": [],
                    "eventType": "post",
                    "postProcess": {},
                },
                "parameters": {
                    "buttonType": "form-submit",
                    "buttonValue": "SIGNON",
                },
                "eventName": "continue",
            },
        ) as response:
            assert response.status == 200
            response = await response.json()
            print("third post completed, interactionToken received")

        async with self.session.post(
            AS_RESUME_URL,
            data={
                "dvResponse": response["dvResponse"],
                "state": new_state,
            },
            allow_redirects=False,
        ) as response:
            assert response.status in (200, 302)

            location = response.headers["Location"]
            auth_code = re.search(r"[?&]code=([^&]+)", location).group(1)

        async with self.session.post(
            AS_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "redirect_uri": REDIRECT_URI,
                "code": auth_code,
                "code_verifier": code_verifier,
                "client_id": "bfcbaf69-aade-4c1b-8f00-c1cb8a193030",
            },
        ) as response:
            assert response.status == 200
            response = await response.json()
            access_token = response["access_token"]
            user = await utils.fetch(
                self.session, API_ME, headers=utils.headers_access(access_token)
            )
            self.user = User(user, self.session, access_token)
            return self.session
