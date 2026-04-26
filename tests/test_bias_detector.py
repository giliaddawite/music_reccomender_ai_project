"""
Tests for the bias detector — three check functions plus the combined runner.

Each check has a "should warn" and "should not warn" case at minimum,
plus edge cases where relevant.
"""

import pytest

from src.bias_detector import (
    check_feature_dominance,
    check_diversity,
    check_score_spread,
    run_bias_check,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def make_song(
    genre="pop",
    mood="happy",
    energy=0.5,
    valence=0.5,
    danceability=0.5,
    acousticness=0.5,
    title="Test Song",
    artist="Test Artist",
    tempo_bpm=120.0,
):
    return {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "acousticness": acousticness,
        "title": title,
        "artist": artist,
        "tempo_bpm": tempo_bpm,
    }


def make_recs(song_list, scores=None):
    """
    Turn a list of song dicts into the (song, score, explanation) format
    that the bias detector expects. Scores default to descending from 0.9.
    """
    if scores is None:
        scores = [0.9 - i * 0.05 for i in range(len(song_list))]
    return [(song, score, "test explanation") for song, score in zip(song_list, scores)]


EQUAL_WEIGHTS = {
    "genre": 0.17,
    "mood": 0.17,
    "energy": 0.17,
    "valence": 0.17,
    "danceability": 0.16,
    "acousticness": 0.16,
}

USER_PREFS = {
    "genre": "pop",
    "mood": "happy",
    "energy": 0.8,
    "valence": 0.7,
    "danceability": 0.6,
    "likes_acoustic": False,
}


# ══════════════════════════════════════════════
# check_feature_dominance
# ══════════════════════════════════════════════

class TestFeatureDominance:

    def test_warns_when_genre_dominates(self):
        """All songs match genre and mood but have random numeric features,
        so categorical matches carry most of the score."""
        songs = [
            make_song(genre="pop", mood="happy", energy=0.1, valence=0.1, danceability=0.1),
            make_song(genre="pop", mood="happy", energy=0.2, valence=0.1, danceability=0.2),
            make_song(genre="pop", mood="happy", energy=0.15, valence=0.15, danceability=0.1),
        ]
        heavy_genre_weights = {
            "genre": 0.40,
            "mood": 0.10,
            "energy": 0.10,
            "valence": 0.15,
            "danceability": 0.13,
            "acousticness": 0.12,
        }

        recs = make_recs(songs)
        warnings = check_feature_dominance(recs, USER_PREFS, heavy_genre_weights)

        assert any("genre" in w.lower() and "dominance" in w.lower() for w in warnings)

    def test_no_warning_with_balanced_weights(self):
        """When weights are equal and songs vary, no single feature should dominate."""
        songs = [
            make_song(genre="pop", mood="happy", energy=0.8, valence=0.7, danceability=0.6),
            make_song(genre="rock", mood="chill", energy=0.7, valence=0.6, danceability=0.5),
            make_song(genre="jazz", mood="intense", energy=0.6, valence=0.5, danceability=0.7),
        ]

        recs = make_recs(songs)
        warnings = check_feature_dominance(recs, USER_PREFS, EQUAL_WEIGHTS)

        dominance_warnings = [w for w in warnings if "dominance" in w.lower()]
        assert len(dominance_warnings) == 0

    def test_empty_recommendations_no_crash(self):
        warnings = check_feature_dominance([], USER_PREFS, EQUAL_WEIGHTS)
        assert warnings == []


# ══════════════════════════════════════════════
# check_diversity
# ══════════════════════════════════════════════

class TestDiversity:

    def test_warns_when_all_same_genre(self):
        """5 pop songs should trigger a genre diversity warning."""
        songs = [make_song(genre="pop") for _ in range(5)]

        recs = make_recs(songs)
        warnings = check_diversity(recs)

        assert any("genre diversity" in w.lower() for w in warnings)

    def test_warns_when_all_same_mood(self):
        """5 happy songs across different genres should trigger a mood warning."""
        songs = [
            make_song(genre="pop", mood="happy"),
            make_song(genre="rock", mood="happy"),
            make_song(genre="jazz", mood="happy"),
            make_song(genre="lofi", mood="happy"),
            make_song(genre="classical", mood="happy"),
        ]

        recs = make_recs(songs)
        warnings = check_diversity(recs)

        assert any("mood diversity" in w.lower() for w in warnings)
        # Genre should be fine since they're all different
        genre_warnings = [w for w in warnings if "genre diversity" in w.lower()]
        assert len(genre_warnings) == 0

    def test_no_warning_with_mixed_results(self):
        """Diverse genres and moods should produce no diversity warnings."""
        songs = [
            make_song(genre="pop", mood="happy"),
            make_song(genre="rock", mood="intense"),
            make_song(genre="jazz", mood="chill"),
            make_song(genre="lofi", mood="relaxed"),
            make_song(genre="classical", mood="peaceful"),
        ]

        recs = make_recs(songs)
        warnings = check_diversity(recs)

        assert len(warnings) == 0

    def test_threshold_boundary_four_out_of_five(self):
        """4/5 = 80% which is exactly at the threshold — should trigger."""
        songs = [
            make_song(genre="pop"),
            make_song(genre="pop"),
            make_song(genre="pop"),
            make_song(genre="pop"),
            make_song(genre="rock"),
        ]

        recs = make_recs(songs)
        warnings = check_diversity(recs)

        assert any("genre diversity" in w.lower() for w in warnings)

    def test_below_threshold_three_out_of_five(self):
        """3/5 = 60% which is below 80% — should not trigger."""
        songs = [
            make_song(genre="pop"),
            make_song(genre="pop"),
            make_song(genre="pop"),
            make_song(genre="rock"),
            make_song(genre="jazz"),
        ]

        recs = make_recs(songs)
        warnings = check_diversity(recs)

        genre_warnings = [w for w in warnings if "genre diversity" in w.lower()]
        assert len(genre_warnings) == 0

    def test_single_recommendation_no_crash(self):
        recs = make_recs([make_song()])
        warnings = check_diversity(recs)
        assert warnings == []

    def test_empty_recommendations_no_crash(self):
        warnings = check_diversity([])
        assert warnings == []


# ══════════════════════════════════════════════
# check_score_spread
# ══════════════════════════════════════════════

class TestScoreSpread:

    def test_warns_when_scores_are_clustered(self):
        """Scores within 0.05 of each other should trigger low confidence."""
        songs = [make_song() for _ in range(5)]
        recs = make_recs(songs, scores=[0.60, 0.59, 0.58, 0.57, 0.56])

        warnings = check_score_spread(recs)

        assert any("low confidence" in w.lower() for w in warnings)

    def test_no_warning_with_clear_spread(self):
        """Big gap between top and bottom means confident ranking."""
        songs = [make_song() for _ in range(5)]
        recs = make_recs(songs, scores=[0.92, 0.85, 0.73, 0.65, 0.58])

        warnings = check_score_spread(recs)

        assert len(warnings) == 0

    def test_above_threshold_no_warning(self):
        """Spread of 0.15 is clearly above the 0.10 threshold — should not trigger.
        Note: 0.70 - 0.60 = 0.09999... in float, so avoid exact-boundary scores."""
        songs = [make_song() for _ in range(3)]
        recs = make_recs(songs, scores=[0.75, 0.65, 0.60])

        warnings = check_score_spread(recs)

        assert len(warnings) == 0

    def test_just_below_threshold(self):
        """Spread of 0.09 should trigger."""
        songs = [make_song() for _ in range(3)]
        recs = make_recs(songs, scores=[0.65, 0.62, 0.56])

        warnings = check_score_spread(recs)

        # 0.65 - 0.56 = 0.09 which is below 0.10
        assert any("low confidence" in w.lower() for w in warnings)

    def test_single_recommendation_no_crash(self):
        recs = make_recs([make_song()], scores=[0.75])
        warnings = check_score_spread(recs)
        assert warnings == []

    def test_empty_recommendations_no_crash(self):
        warnings = check_score_spread([])
        assert warnings == []


# ══════════════════════════════════════════════
# run_bias_check (combined runner)
# ══════════════════════════════════════════════

class TestRunBiasCheck:

    def test_returns_warnings_from_all_checks(self):
        """Construct a scenario that triggers all three checks at once."""
        # All same genre + mood → diversity warning
        # Clustered scores → spread warning
        # Heavy genre weight + all genre matches → dominance warning
        songs = [
            make_song(genre="pop", mood="happy", energy=0.1, valence=0.1, danceability=0.1),
            make_song(genre="pop", mood="happy", energy=0.12, valence=0.11, danceability=0.1),
            make_song(genre="pop", mood="happy", energy=0.11, valence=0.1, danceability=0.12),
            make_song(genre="pop", mood="happy", energy=0.13, valence=0.12, danceability=0.11),
            make_song(genre="pop", mood="happy", energy=0.1, valence=0.13, danceability=0.1),
        ]
        heavy_genre_weights = {
            "genre": 0.40,
            "mood": 0.10,
            "energy": 0.10,
            "valence": 0.15,
            "danceability": 0.13,
            "acousticness": 0.12,
        }
        recs = make_recs(songs, scores=[0.62, 0.61, 0.60, 0.59, 0.58])

        warnings = run_bias_check(recs, USER_PREFS, heavy_genre_weights)

        # Should have at least one warning from each check
        assert any("dominance" in w.lower() for w in warnings)
        assert any("diversity" in w.lower() for w in warnings)
        assert any("confidence" in w.lower() for w in warnings)

    def test_returns_empty_when_everything_is_healthy(self):
        """Diverse results, spread scores, balanced weights → no warnings."""
        songs = [
            make_song(genre="pop", mood="happy", energy=0.8, valence=0.7, danceability=0.6),
            make_song(genre="rock", mood="intense", energy=0.7, valence=0.5, danceability=0.4),
            make_song(genre="jazz", mood="chill", energy=0.4, valence=0.6, danceability=0.3),
            make_song(genre="lofi", mood="relaxed", energy=0.3, valence=0.8, danceability=0.5),
            make_song(genre="classical", mood="peaceful", energy=0.2, valence=0.4, danceability=0.2),
        ]
        recs = make_recs(songs, scores=[0.90, 0.78, 0.65, 0.55, 0.42])

        warnings = run_bias_check(recs, USER_PREFS, EQUAL_WEIGHTS)

        assert warnings == []
