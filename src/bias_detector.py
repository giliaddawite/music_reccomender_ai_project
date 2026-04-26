"""Bias detector for the music recommender.

Watches the scoring engine's output and raises warnings when something
looks off. Does NOT fix anything — that's the feedback agent's job.
This module just flags problems so they're visible in the demo and log.
"""

from typing import Dict, List, Tuple

from src.recommender import score_song


# ──────────────────────────────────────────────
# Thresholds (tune these if warnings are too noisy or too quiet)
# ──────────────────────────────────────────────

# If one feature contributes more than this fraction of the total score, flag it
DOMINANCE_THRESHOLD = 0.45

# If this fraction or more of the top results share the same genre or mood, flag it
DIVERSITY_THRESHOLD = 0.80

# If the gap between the highest and lowest score in the top results is below this, flag it
SPREAD_THRESHOLD = 0.10


def check_feature_dominance(
    recommendations: List[Tuple[Dict, float, str]],
    user_prefs: Dict,
    weights: Dict,
) -> List[str]:
    """Check if one feature is doing all the heavy lifting."""
    warnings = []

    feature_totals = {f: 0.0 for f in weights}
    song_count = len(recommendations)

    if song_count == 0:
        return warnings

    for song, score, _ in recommendations:
        contributions = _get_feature_contributions(song, user_prefs, weights)
        for feature, value in contributions.items():
            feature_totals[feature] += value

    feature_avgs = {f: total / song_count for f, total in feature_totals.items()}
    avg_total = sum(feature_avgs.values())
    if avg_total == 0:
        return warnings

    for feature, avg_contribution in feature_avgs.items():
        fraction = avg_contribution / avg_total
        if fraction > DOMINANCE_THRESHOLD:
            warnings.append(
                f"⚠ {feature} dominance: contributed {fraction:.0%} of the average "
                f"score across top results (threshold: {DOMINANCE_THRESHOLD:.0%})"
            )

    return warnings


def check_diversity(recommendations: List[Tuple[Dict, float, str]]) -> List[str]:
    """Check if all top results look the same."""
    warnings = []
    song_count = len(recommendations)

    if song_count < 2:
        return warnings

    genres = [song["genre"] for song, _, _ in recommendations]
    moods = [song["mood"] for song, _, _ in recommendations]

    top_genre = max(set(genres), key=genres.count)
    top_mood = max(set(moods), key=moods.count)

    genre_fraction = genres.count(top_genre) / song_count
    mood_fraction = moods.count(top_mood) / song_count

    if genre_fraction >= DIVERSITY_THRESHOLD:
        warnings.append(
            f"⚠ Low genre diversity: {genre_fraction:.0%} of results are {top_genre} "
            f"({genres.count(top_genre)}/{song_count} songs)"
        )

    if mood_fraction >= DIVERSITY_THRESHOLD:
        warnings.append(
            f"⚠ Low mood diversity: {mood_fraction:.0%} of results are {top_mood} "
            f"({moods.count(top_mood)}/{song_count} songs)"
        )

    return warnings


def check_score_spread(recommendations: List[Tuple[Dict, float, str]]) -> List[str]:
    """Check if the system is confident in its rankings."""
    warnings = []

    if len(recommendations) < 2:
        return warnings

    scores = [score for _, score, _ in recommendations]
    spread = max(scores) - min(scores)

    if spread < SPREAD_THRESHOLD:
        warnings.append(
            f"⚠ Low confidence: score spread is only {spread:.3f} "
            f"(top={max(scores):.3f}, bottom={min(scores):.3f}, "
            f"threshold: {SPREAD_THRESHOLD})"
        )

    return warnings


def run_bias_check(
    recommendations: List[Tuple[Dict, float, str]],
    user_prefs: Dict,
    weights: Dict,
) -> List[str]:
    """Run all three bias checks and return a combined list of warnings."""
    warnings = []
    warnings.extend(check_feature_dominance(recommendations, user_prefs, weights))
    warnings.extend(check_diversity(recommendations))
    warnings.extend(check_score_spread(recommendations))
    return warnings


def _get_feature_contributions(song: Dict, user_prefs: Dict, weights: Dict) -> Dict[str, float]:
    """Break down a song's score into per-feature contributions."""
    contributions = {}

    contributions["genre"] = weights["genre"] if song["genre"] == user_prefs.get("genre") else 0.0
    contributions["mood"] = weights["mood"] if song["mood"] == user_prefs.get("mood") else 0.0

    energy_sim = 1.0 - abs(song["energy"] - user_prefs.get("energy", 0.5))
    contributions["energy"] = weights["energy"] * energy_sim

    valence_sim = 1.0 - abs(song["valence"] - user_prefs.get("valence", 0.5))
    contributions["valence"] = weights["valence"] * valence_sim

    dance_sim = 1.0 - abs(song["danceability"] - user_prefs.get("danceability", 0.5))
    contributions["danceability"] = weights["danceability"] * dance_sim

    acoust_val = (
        song["acousticness"]
        if user_prefs.get("likes_acoustic", False)
        else (1.0 - song["acousticness"])
    )
    contributions["acousticness"] = weights["acousticness"] * acoust_val

    return contributions
