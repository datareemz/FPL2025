from constants import endpoints
import utils
import aiohttp


API_USER_TEAM = endpoints["API"]["USER_TEAM"]
API_ME = endpoints["API"]["ME"]
API_MANAGER_INFO_PER_GW_URL = endpoints["API"]["MANAGER_INFO_PER_GW"]
API_MANAGER_INFO = endpoints["API"]["MANAGER_INFO"]



class User:
    def __init__(self, user_information, session, access_token):
        self.session = session
        self.access_token = access_token
        for k, v in user_information["player"].items():
            setattr(self, k, v)

    def logged_in(self):
        """Checks that the user is logged in within the session.

        :return: True if user is logged in else False
        :rtype: bool
        """
        assert all(
            x
            in self.session.cookie_jar.filter_cookies(
                "https://account.premierleague.com/"
            )
            for x in ["interactionToken", "interactionId"]
        ), "Must Be logged in"
        return True

    async def get_current_user_entry(self):
        return getattr(self, "entry", None)

    # async def get_user(self, user_id=None):
    #     """Returns the user with the given ``user_id``
    #     :param session:
    #     :param user_id: A user's ID.
    #     :type user_id: string or int
    #     :rtype: :class:`User`
    #     """
    #     if user_id:
    #         assert int(user_id) > 0, "User ID must be a positive number."
    #     else:
    #         try:
    #             user = await self.get_current_user()
    #             return user
    #         except TypeError:
    #             raise Exception(
    #                 "You must log in before using `get_user` if "
    #                 "you do not provide a user ID."
    #             )

    async def get_manager_info_for_gw(self, user, gw):
        """Returns info on the managers team per gameweek. Requires the user to have
        logged in using ``fpl.login()``.

        :rtype: list
        """
        if not self.logged_in():
            raise Exception("User must be logged in.")
        try:
            response = await utils.fetch(
                self.session, API_MANAGER_INFO_PER_GW_URL.format(f=user.entry, gw=gw)
            )
        except Exception as e:
            raise Exception("Client has not set a team for gameweek " + str(gw))
        return response["picks"]

    async def get_manager_info(self):
        """Returns info on the managers team. Requires the user to have
        logged in using ``fpl.login()``.
        :rtype: list
        """
        if not self.logged_in():
            raise Exception("User must be logged in.")
        try:
            response = await utils.fetch(self.session, API_MANAGER_INFO.format(f=getattr(self, "entry", None)))
        except Exception as e:
            raise Exception("Client has not set a team for gameweek ")
        return response

    async def get_users_team_info(self):
        """Returns a logged-in user's current team. Requires the user to have
        logged in using ``fpl.login()``.
        :rtype: list
        """
        if not self.logged_in():
            raise Exception("User must be logged in.")
        try:
            response = await utils.fetch(
                self.session,
                API_USER_TEAM.format(f=getattr(self, "entry", None)),
                headers=utils.headers_access(self.access_token),
            )
        except Exception as e:
            raise Exception("Client has not set a team for gameweek ")
        return response

    async def get_transfers_status(self):
        """Returns a logged in user's transfer status, which is a dictionary
        containing their bank value, how many free transfers they have left
        and so on. Requires the user to have logged in using ``fpl.login()``.

        Information is taken from e.g.:
            https://fantasy.premierleague.com/api/my-team/81629336/

        :rtype: dict
        """
        if not self.logged_in():
            raise Exception("User must be logged in.")
        try:
            response = await utils.fetch(self.session, API_USER_TEAM.format(f=getattr(self, "entry", None)),
                                         headers=utils.headers_access(self.access_token))
        except aiohttp.client_exceptions.ClientResponseError:
            raise Exception("User ID does not match provided email address!")
        return response["transfers"]