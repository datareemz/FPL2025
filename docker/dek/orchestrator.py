import os
import asyncio
import logging
from datetime import datetime, timezone

import aiohttp
import pandas as pd
from pycaret.regression import predict_model

import utils
from api.FPL import FPL
from api.FPL_helpers import FPLHelpers
from auth.fpl_auth import FPLAuth
from constants import FPL_BUCKET
from dataModel.player import Player

# --- Configuration ---
AUTO_TRANSFER = os.getenv("FPL_AUTO_TRANSFER", "false").lower() == "true"
MAX_TRANSFERS_PER_RUN = int(os.getenv("FPL_MAX_TRANSFERS", "2"))
MODEL_GCS_BLOB = "ml/Final et Model ROI-Target - Retrained.pkl"
MODEL_LOCAL_PATH = "ml/Final et Model ROI-Target - Retrained"
PERFORMANCE_BLOB = "performance/weekly_performance.csv"
METRICS = [
    "total_points",
    "goals_scored",
    "assists",
    "minutes",
    "starts",
    "roi",
    "points_per_game",
]

logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


async def initialize_session():
    """Create aiohttp session, authenticate, and return (session, fpl)."""
    session = aiohttp.ClientSession(trust_env=True)
    auth = FPLAuth(session)
    helpers = FPLHelpers(session)
    fpl = FPL(session, auth, helpers)
    await fpl.login()
    return session, fpl


def load_ml_model():
    """Load the ML model from GCS (preferred) or local fallback."""
    model = utils.load_model_from_gcs(FPL_BUCKET, MODEL_GCS_BLOB, MODEL_LOCAL_PATH)
    logger.info("ROI model loaded")
    return model


async def prepare_and_predict(fpl, model, game_week):
    """Fetch data for the game week, run predictions, and return top Player objects."""
    await asyncio.wait_for(fpl.helpers.get_data(game_week), timeout=60.0)

    unseen_data = pd.read_csv(f"datastore/current/FPL_data_{game_week}.csv")
    predictions = predict_model(model, data=unseen_data).sort_values(
        "prediction_label", ascending=False
    )

    predictions = predictions.head(250)
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

    player_objects = []
    for record in predicted_players:
        player = await fpl.get_current_player(player=record, convert_hist=False)
        player_objects.append(player)

    return player_objects


async def get_current_team(fpl):
    """Fetch the user's current squad and return sorted Player objects."""
    user = await fpl.get_user()
    my_players = await fpl.get_users_players(user)

    team = []
    for pick in my_players:
        player = await fpl.get_current_player(player_id=pick["element"])
        team.append(player)

    team.sort(key=lambda x: x.roi())
    return team


def find_transfer_recommendations(fpl, team, player_pool):
    """Analyze the team and find replacement candidates for weakest players."""
    analysis = fpl.helpers.get_team_analysis(team, metrics=METRICS)
    weakest = analysis["no_position"]
    metric_stats = analysis["metric_stats"]

    recommendations = []
    for weak_player in weakest[:4]:
        candidates = fpl.helpers.find_valid_replacement(
            player_out=weak_player,
            player_pool=player_pool,
            current_team=team,
            metrics=METRICS,
            metric_stats=metric_stats,
        )
        if candidates:
            recommendations.append(
                {"weak_player": weak_player, "candidates": candidates}
            )
            logger.info(
                "Weak player: %s | %d candidates found",
                weak_player["player_name"],
                len(candidates),
            )
        else:
            logger.info(
                "Weak player: %s | No valid replacements found",
                weak_player["player_name"],
            )

    return recommendations


async def execute_or_log_transfers(fpl, recommendations):
    """Execute transfers if AUTO_TRANSFER is enabled, otherwise just log them.

    Returns a list of dicts describing transfers made (or recommended).
    """
    transfer_log = []

    if not recommendations:
        logger.info("No transfer recommendations to process.")
        return transfer_log

    status = await fpl.get_transfers_status()
    free_transfers = status["limit"] - status["made"]
    bank = status["value"]
    logger.info("Free transfers: %d | Bank: %.1f", free_transfers, bank)

    transfers_to_make = min(free_transfers, MAX_TRANSFERS_PER_RUN, len(recommendations))

    for i in range(transfers_to_make):
        rec = recommendations[i]
        weak = rec["weak_player"]
        best = rec["candidates"][0]

        entry = {
            "out": weak["player_name"],
            "in": str(best["player"]),
            "improvement": best["total_metric_improvement_val"],
            "cost_diff": best["cost_diff"],
        }

        if AUTO_TRANSFER:
            try:
                out_id = weak["player_object"].id
                in_id = best["player"]["id"]
                await fpl.transfer([out_id], [in_id], max_hit=0)
                entry["status"] = "executed"
                logger.info("Transfer executed: %s -> %s", entry["out"], entry["in"])
            except Exception as e:
                entry["status"] = f"failed: {e}"
                logger.error("Transfer failed: %s", e)
        else:
            entry["status"] = "recommended"
            logger.info(
                "Transfer recommended: %s -> %s (improvement: %.2f)",
                entry["out"],
                entry["in"],
                entry["improvement"],
            )

        transfer_log.append(entry)

    return transfer_log


async def track_performance(fpl, game_week):
    """Fetch and persist performance for all game weeks up to game_week.

    Reads CSV once from GCS, skips tracked game weeks, appends new rows,
    writes once. Max 2 GCS calls per invocation.

    Args:
        fpl: Authenticated FPL instance.
        game_week: Current/upcoming game week (tracks 1 through game_week - 1).

    Returns:
        Number of new game weeks appended.
    """
    if game_week <= 0:
        logger.info("Invalid game week: %d, skipping.", game_week)
        return 0

    df = utils.read_performance_from_gcs(FPL_BUCKET, PERFORMANCE_BLOB)
    tracked = set(df["game_week"].values) if not df.empty else set()

    new_rows = []
    for gw in range(1, game_week):
        if gw in tracked:
            logger.info("GW%d already tracked, skipping.", gw)
            continue
        row = await fpl.get_performance_for_game_week(gw)
        if row is None:
            logger.warning("No data returned for GW%d, skipping.", gw)
            continue
        new_rows.append(row)

    if not new_rows:
        logger.info("No new game weeks to track.")
        return 0

    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    utils.write_performance_to_gcs(df, FPL_BUCKET, PERFORMANCE_BLOB)
    logger.info("Appended %d new game weeks to performance CSV.", len(new_rows))
    return len(new_rows)


async def main():
    configure_logging()
    logger.info(
        "FPL Bot starting | AUTO_TRANSFER=%s | MAX_TRANSFERS=%d",
        AUTO_TRANSFER,
        MAX_TRANSFERS_PER_RUN,
    )

    session = None
    try:
        # Phase 1: Initialize
        model = load_ml_model()
        session, fpl = await initialize_session()

        # Phase 2: Get game week
        game_week = await fpl.helpers.get_upcoming_gameweek()
        logger.info("Upcoming game week: %d", game_week)

        # Phase 3: Predict
        player_pool = await prepare_and_predict(fpl, model, game_week)

        # Phase 4: Analyze current team
        team = await get_current_team(fpl)

        # Phase 5: Find transfer recommendations
        recommendations = find_transfer_recommendations(fpl, team, player_pool)

        # Phase 6: Execute or log transfers
        transfer_log = await execute_or_log_transfers(fpl, recommendations)

        # Phase 7: Track performance (non-fatal)
        try:
            await track_performance(fpl, game_week)
        except Exception as e:
            logger.warning("Performance tracking failed (non-fatal): %s", e)

        logger.info("FPL Bot workflow complete.")
    except Exception as e:
        logger.error("FPL Bot error: %s", e)
        raise
    finally:
        if session:
            await session.close()


if __name__ == "__main__":
    asyncio.run(main())
