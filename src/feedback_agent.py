from typing import List, Dict, Tuple
from recommender import recommend_songs, DEFAULT_WEIGHTS
from logger import log_decision
from bias_detector import run_bias_check

RATING_OPTIONS = {"l": "like", "d": "dislike", "s": "skip"}
NUDGE = 0.05
WEIGHT_MIN = 0.05  # floor — no feature is ever fully ignored
WEIGHT_MAX = 0.40  # ceiling — no single feature can dominate


def collect_feedback(recommendations: List[Tuple[Dict, float, str]]) -> List[Dict]:
    """
    Prompt the user to rate each recommendation.
    Returns a list of {"song": song_dict, "rating": "like"|"dislike"|"skip"}.
    """
    feedback = []
    print("\nRate each song — (l)ike / (d)islike / (s)kip:")
    for song, score, _ in recommendations:
        while True:
            raw = input(f"  {song['title']} by {song['artist']} [{score:.2f}]: ").strip().lower()
            if raw in RATING_OPTIONS:
                feedback.append({"song": song, "rating": RATING_OPTIONS[raw]})
                break
            print("    Enter l, d, or s.")
    return feedback


def analyze_feedback(feedback_list: List[Dict]) -> Dict:
    """
    Split ratings into liked / disliked groups and compare their feature profiles.

    Patterns returned:
      liked_genres        set of genres in liked songs
      disliked_genres     set of genres in disliked songs
      genre_diverse       True when liked songs span > 1 genre
      avg_liked_energy    mean energy of liked songs (None if no likes)
      avg_disliked_energy mean energy of disliked songs (None if no dislikes)
      energy_gap          |avg_liked - avg_disliked| energy (None if either empty)
      avg_liked_valence / avg_liked_danceability / avg_liked_acousticness
      like_count / dislike_count / skip_count
    """
    liked = [f["song"] for f in feedback_list if f["rating"] == "like"]
    disliked = [f["song"] for f in feedback_list if f["rating"] == "dislike"]

    def avg(songs, key):
        return sum(s[key] for s in songs) / len(songs) if songs else None

    def gap(a_liked, a_disliked):
        if a_liked is not None and a_disliked is not None:
            return abs(a_liked - a_disliked)
        return None

    al_energy = avg(liked, "energy")
    ad_energy = avg(disliked, "energy")
    al_valence = avg(liked, "valence")
    ad_valence = avg(disliked, "valence")
    al_dance = avg(liked, "danceability")
    ad_dance = avg(disliked, "danceability")
    al_ac = avg(liked, "acousticness")
    ad_ac = avg(disliked, "acousticness")

    return {
        "liked_genres": {s["genre"] for s in liked},
        "disliked_genres": {s["genre"] for s in disliked},
        "genre_diverse": len({s["genre"] for s in liked}) > 1,
        "avg_liked_energy": al_energy,
        "avg_disliked_energy": ad_energy,
        "energy_gap": gap(al_energy, ad_energy),
        "avg_liked_valence": al_valence,
        "avg_disliked_valence": ad_valence,
        "valence_gap": gap(al_valence, ad_valence),
        "avg_liked_danceability": al_dance,
        "avg_disliked_danceability": ad_dance,
        "danceability_gap": gap(al_dance, ad_dance),
        "avg_liked_acousticness": al_ac,
        "avg_disliked_acousticness": ad_ac,
        "acousticness_gap": gap(al_ac, ad_ac),
        "like_count": len(liked),
        "dislike_count": len(disliked),
        "skip_count": len(feedback_list) - len(liked) - len(disliked),
    }


def adjust_weights(current_weights: Dict, patterns: Dict) -> Tuple[Dict, str]:
    """
    Nudge each weight by NUDGE (0.05) based on feedback patterns,
    clamp to [WEIGHT_MIN, WEIGHT_MAX], then normalize so weights sum to 1.0.
    Returns (new_weights, reasoning).
    """
    w = dict(current_weights)  # copy — don't mutate the caller's dict
    reasons = []

    if patterns["genre_diverse"]:
        w["genre"] -= NUDGE
        reasons.append("reduced genre (liked songs span multiple genres)")
    elif len(patterns["liked_genres"]) == 1 and patterns["like_count"] >= 2:
        w["genre"] += NUDGE
        reasons.append(f"boosted genre (consistent {next(iter(patterns['liked_genres']))} preference)")

    gap = patterns["energy_gap"]
    if gap is not None:
        if gap > 0.20:
            w["energy"] += NUDGE
            reasons.append(
                f"boosted energy (liked ~{patterns['avg_liked_energy']:.2f}, "
                f"disliked ~{patterns['avg_disliked_energy']:.2f})"
            )
        else:
            w["energy"] -= NUDGE
            reasons.append("reduced energy (liked/disliked energy too similar)")

    val_gap = patterns["valence_gap"]
    if val_gap is not None:
        if val_gap > 0.20:
            w["valence"] += NUDGE
            reasons.append(
                f"boosted valence (liked ~{patterns['avg_liked_valence']:.2f}, "
                f"disliked ~{patterns['avg_disliked_valence']:.2f})"
            )
        else:
            w["valence"] -= NUDGE
            reasons.append("reduced valence (liked/disliked valence too similar)")

    dance_gap = patterns["danceability_gap"]
    if dance_gap is not None:
        if dance_gap > 0.20:
            w["danceability"] += NUDGE
            reasons.append(
                f"boosted danceability (liked ~{patterns['avg_liked_danceability']:.2f}, "
                f"disliked ~{patterns['avg_disliked_danceability']:.2f})"
            )
        else:
            w["danceability"] -= NUDGE
            reasons.append("reduced danceability (liked/disliked danceability too similar)")

    ac_gap = patterns["acousticness_gap"]
    if ac_gap is not None:
        if ac_gap > 0.20:
            w["acousticness"] += NUDGE
            reasons.append(
                f"boosted acousticness (liked ~{patterns['avg_liked_acousticness']:.2f}, "
                f"disliked ~{patterns['avg_disliked_acousticness']:.2f})"
            )
        else:
            w["acousticness"] -= NUDGE
            reasons.append("reduced acousticness (liked/disliked acousticness too similar)")

    w = {k: max(WEIGHT_MIN, min(WEIGHT_MAX, v)) for k, v in w.items()}

    total = sum(w.values())
    w = {k: round(v / total, 4) for k, v in w.items()}

    reasoning = "; ".join(reasons) if reasons else "no strong patterns detected, weights unchanged"
    return w, reasoning


def run_feedback_loop(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    rounds: int = 3,
    initial_weights: Dict = None,
) -> List[Dict]:
    """
    Orchestrator: recommend → collect feedback → analyze → adjust weights → log → repeat.
    Returns the full log list for model card / demo output.
    """
    weights = dict(initial_weights) if initial_weights else dict(DEFAULT_WEIGHTS)
    full_log = []

    for round_num in range(1, rounds + 1):
        print(f"\n{'=' * 42}")
        print(f"Round {round_num}  |  weights: {_fmt_weights(weights)}")
        print("=" * 42)

        recommendations = recommend_songs(user_prefs, songs, k=k, weights=weights)
        if not recommendations:
            print("No songs available.")
            break

        for song, score, explanation in recommendations:
            print(f"  {song['title']} by {song['artist']}  [{score:.3f}]")
            print(f"    {explanation}\n")

        bias_warnings = run_bias_check(recommendations, user_prefs, weights)
        if bias_warnings:
            print("\n Bias check:")
            for warning in bias_warnings:
                print(f"  {warning}")

        feedback = collect_feedback(recommendations)
        patterns = analyze_feedback(feedback)
        new_weights, reasoning = adjust_weights(weights, patterns)

        entry = {
            "round": round_num,
            "weights_before": dict(weights),
            "weights_after": new_weights,
            "reasoning": reasoning,
            "bias_warnings": bias_warnings,
            "like_count": patterns["like_count"],
            "dislike_count": patterns["dislike_count"],
            "skip_count": patterns["skip_count"],
        }
        log_decision(entry)
        full_log.append(entry)

        print(f"\nWeight update: {reasoning}")
        weights = new_weights

        if round_num < rounds:
            cont = input("\nContinue to next round? (y/n): ").strip().lower()
            if cont != "y":
                break

    print("\nSession complete. Decisions logged to data/decisions.log")
    return full_log


def _fmt_weights(w: Dict) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in w.items())
