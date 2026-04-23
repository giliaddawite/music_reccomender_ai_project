"""
Tests for the feedback agent — three layers:
 
Layer 1: analyze_feedback computes correct measurements
Layer 2: adjust_weights makes correct decisions based on patterns
Layer 3: multi-round integration (fake user proves the system learns)
"""
 
import pytest
 
from src.feedback_agent import analyze_feedback, adjust_weights
from src.recommender import DEFAULT_WEIGHTS, recommend_songs
 
 
# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────
 
def make_song(
    genre: str = "pop",
    energy: float = 0.5,
    valence: float = 0.5,
    danceability: float = 0.5,
    acousticness: float = 0.5,
    title: str = "Test Song",
    artist: str = "Test Artist",
    mood: str = "happy",
    tempo_bpm: float = 120.0,
):
    """Build a song dict with sensible defaults so each test only sets what matters."""
    return {
        "genre": genre,
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "acousticness": acousticness,
        "title": title,
        "artist": artist,
        "mood": mood,
        "tempo_bpm": tempo_bpm,
    }
 
 
def make_feedback(songs_and_ratings):
    """Shortcut: pass a list of (song_dict, rating_str) tuples."""
    return [{"song": song, "rating": rating} for song, rating in songs_and_ratings]
 
 
# ══════════════════════════════════════════════
# LAYER 1 — analyze_feedback measurements
# ══════════════════════════════════════════════
 
class TestAnalyzeFeedbackMeasurements:
 
    def test_energy_gap_calculates_expected(self):
        feedback = make_feedback(
            [(make_song(energy=0.9), "like")] * 3
            + [(make_song(genre="rock", energy=0.3), "dislike")] * 2
        )
 
        patterns = analyze_feedback(feedback)
 
        assert patterns["avg_liked_energy"] == pytest.approx(0.9)
        assert patterns["avg_disliked_energy"] == pytest.approx(0.3)
        assert patterns["energy_gap"] == pytest.approx(0.6)
        assert patterns["like_count"] == 3
        assert patterns["dislike_count"] == 2
        assert patterns["skip_count"] == 0
 
    def test_genre_diverse_true_when_liked_genres_vary(self):
        feedback = make_feedback([
            (make_song(genre="pop", energy=0.7), "like"),
            (make_song(genre="pop", energy=0.6), "like"),
            (make_song(genre="rock", energy=0.8), "like"),
            (make_song(genre="jazz", energy=0.4), "dislike"),
        ])
 
        patterns = analyze_feedback(feedback)
 
        assert patterns["genre_diverse"] is True
        assert patterns["liked_genres"] == {"pop", "rock"}
        assert patterns["disliked_genres"] == {"jazz"}
 
    def test_genre_diverse_false_when_all_likes_same_genre(self):
        feedback = make_feedback([
            (make_song(genre="pop", energy=0.7), "like"),
            (make_song(genre="pop", energy=0.8), "like"),
            (make_song(genre="rock", energy=0.3), "dislike"),
        ])
 
        patterns = analyze_feedback(feedback)
 
        assert patterns["genre_diverse"] is False
        assert patterns["liked_genres"] == {"pop"}
 
    def test_all_skips_returns_none_for_averages(self):
        feedback = make_feedback([
            (make_song(energy=0.7), "skip"),
            (make_song(energy=0.4), "skip"),
        ])
 
        patterns = analyze_feedback(feedback)
 
        assert patterns["avg_liked_energy"] is None
        assert patterns["avg_disliked_energy"] is None
        assert patterns["energy_gap"] is None
        assert patterns["like_count"] == 0
        assert patterns["dislike_count"] == 0
        assert patterns["skip_count"] == 2
 
    def test_likes_with_no_dislikes_returns_none_for_gaps(self):
        """Common scenario: user likes 3 songs and skips the rest."""
        feedback = make_feedback([
            (make_song(energy=0.9, valence=0.8), "like"),
            (make_song(energy=0.7, valence=0.6), "like"),
            (make_song(energy=0.8, valence=0.7), "like"),
            (make_song(energy=0.5), "skip"),
            (make_song(energy=0.3), "skip"),
        ])
 
        patterns = analyze_feedback(feedback)
 
        # Liked averages should compute normally
        assert patterns["avg_liked_energy"] == pytest.approx(0.8)
        assert patterns["avg_liked_valence"] == pytest.approx(0.7)
        # But gaps are None because there's no disliked group to compare against
        assert patterns["energy_gap"] is None
        assert patterns["valence_gap"] is None
        assert patterns["danceability_gap"] is None
        assert patterns["acousticness_gap"] is None
 
    def test_valence_and_danceability_gaps_compute_correctly(self):
        feedback = make_feedback([
            (make_song(valence=0.9, danceability=0.8), "like"),
            (make_song(valence=0.7, danceability=0.6), "like"),
            (make_song(valence=0.2, danceability=0.2), "dislike"),
        ])
 
        patterns = analyze_feedback(feedback)
 
        assert patterns["avg_liked_valence"] == pytest.approx(0.8)
        assert patterns["avg_disliked_valence"] == pytest.approx(0.2)
        assert patterns["valence_gap"] == pytest.approx(0.6)
        assert patterns["avg_liked_danceability"] == pytest.approx(0.7)
        assert patterns["avg_disliked_danceability"] == pytest.approx(0.2)
        assert patterns["danceability_gap"] == pytest.approx(0.5)
 
 
# ══════════════════════════════════════════════
# LAYER 2 — adjust_weights decisions
# ══════════════════════════════════════════════
 
class TestAdjustWeightsDecisions:
 
    def _base_patterns(self, **overrides):
        """Neutral patterns dict — no gaps trigger any changes by default."""
        base = {
            "liked_genres": {"pop"},
            "disliked_genres": {"rock"},
            "genre_diverse": False,
            "avg_liked_energy": 0.5,
            "avg_disliked_energy": 0.5,
            "energy_gap": 0.0,
            "avg_liked_valence": 0.5,
            "avg_disliked_valence": 0.5,
            "valence_gap": 0.0,
            "avg_liked_danceability": 0.5,
            "avg_disliked_danceability": 0.5,
            "danceability_gap": 0.0,
            "avg_liked_acousticness": 0.5,
            "avg_disliked_acousticness": 0.5,
            "acousticness_gap": 0.0,
            "like_count": 2,
            "dislike_count": 2,
            "skip_count": 1,
        }
        base.update(overrides)
        return base
 
    def test_big_energy_gap_boosts_energy_weight(self):
        patterns = self._base_patterns(
            avg_liked_energy=0.9,
            avg_disliked_energy=0.3,
            energy_gap=0.6,
        )
 
        new_weights, reasoning = adjust_weights(dict(DEFAULT_WEIGHTS), patterns)
 
        # Energy should have increased relative to other features
        assert new_weights["energy"] > DEFAULT_WEIGHTS["energy"]
        assert "boosted energy" in reasoning
 
    def test_small_energy_gap_reduces_energy_weight(self):
        patterns = self._base_patterns(energy_gap=0.05)
 
        new_weights, reasoning = adjust_weights(dict(DEFAULT_WEIGHTS), patterns)
 
        assert new_weights["energy"] < DEFAULT_WEIGHTS["energy"]
        assert "reduced energy" in reasoning
 
    def test_genre_diverse_reduces_genre_weight(self):
        patterns = self._base_patterns(
            genre_diverse=True,
            liked_genres={"pop", "rock", "jazz"},
        )
 
        new_weights, reasoning = adjust_weights(dict(DEFAULT_WEIGHTS), patterns)
 
        assert new_weights["genre"] < DEFAULT_WEIGHTS["genre"]
        assert "reduced genre" in reasoning
 
    def test_consistent_genre_boosts_genre_weight(self):
        patterns = self._base_patterns(
            genre_diverse=False,
            liked_genres={"pop"},
            like_count=3,
        )
 
        new_weights, reasoning = adjust_weights(dict(DEFAULT_WEIGHTS), patterns)
 
        assert new_weights["genre"] > DEFAULT_WEIGHTS["genre"]
        assert "boosted genre" in reasoning
 
    def test_big_valence_gap_boosts_valence_weight(self):
        patterns = self._base_patterns(
            avg_liked_valence=0.9,
            avg_disliked_valence=0.2,
            valence_gap=0.7,
        )
 
        new_weights, reasoning = adjust_weights(dict(DEFAULT_WEIGHTS), patterns)
 
        assert new_weights["valence"] > DEFAULT_WEIGHTS["valence"]
        assert "boosted valence" in reasoning
 
    def test_big_danceability_gap_boosts_danceability_weight(self):
        patterns = self._base_patterns(
            avg_liked_danceability=0.85,
            avg_disliked_danceability=0.2,
            danceability_gap=0.65,
        )
 
        new_weights, reasoning = adjust_weights(dict(DEFAULT_WEIGHTS), patterns)
 
        assert new_weights["danceability"] > DEFAULT_WEIGHTS["danceability"]
        assert "boosted danceability" in reasoning
 
    def test_big_acousticness_gap_boosts_acousticness_weight(self):
        patterns = self._base_patterns(
            avg_liked_acousticness=0.8,
            avg_disliked_acousticness=0.1,
            acousticness_gap=0.7,
        )
 
        new_weights, reasoning = adjust_weights(dict(DEFAULT_WEIGHTS), patterns)
 
        assert new_weights["acousticness"] > DEFAULT_WEIGHTS["acousticness"]
        assert "boosted acousticness" in reasoning
 
    # ── Safety invariants ──
 
    def test_weights_always_sum_to_one(self):
        """No matter what patterns we feed in, weights must sum to 1.0."""
        # Extreme case: every numeric feature has a huge gap, genre is diverse
        patterns = self._base_patterns(
            genre_diverse=True,
            liked_genres={"pop", "rock"},
            energy_gap=0.8,
            valence_gap=0.8,
            danceability_gap=0.8,
            acousticness_gap=0.8,
        )
 
        new_weights, _ = adjust_weights(dict(DEFAULT_WEIGHTS), patterns)
 
        assert sum(new_weights.values()) == pytest.approx(1.0, abs=0.01)
 
    def test_weights_stay_within_bounds(self):
        """No weight should drop below 0.05 or exceed 0.40."""
        # Push everything down by using tiny gaps across 3+ rounds
        weights = dict(DEFAULT_WEIGHTS)
        small_gap_patterns = self._base_patterns(
            energy_gap=0.05,
            valence_gap=0.05,
            danceability_gap=0.05,
            acousticness_gap=0.05,
        )
 
        # Run adjust_weights multiple times to try to force weights to extremes
        for _ in range(10):
            weights, _ = adjust_weights(weights, small_gap_patterns)
 
        for feature, value in weights.items():
            assert value >= 0.04, f"{feature} dropped below floor"  # 0.04 for rounding
            assert value <= 0.41, f"{feature} exceeded ceiling"     # 0.41 for rounding
 
    def test_no_changes_when_all_gaps_are_none(self):
        """If user only liked or only disliked, gaps are None and weights shouldn't crash."""
        patterns = self._base_patterns(
            energy_gap=None,
            valence_gap=None,
            danceability_gap=None,
            acousticness_gap=None,
            avg_liked_energy=0.7,
            avg_disliked_energy=None,
            like_count=3,
            dislike_count=0,
        )
 
        new_weights, reasoning = adjust_weights(dict(DEFAULT_WEIGHTS), patterns)
 
        # Should still return valid weights that sum to 1.0
        assert sum(new_weights.values()) == pytest.approx(1.0, abs=0.01)
 
    def test_original_weights_not_mutated(self):
        """adjust_weights should work on a copy, not modify the input dict."""
        original = dict(DEFAULT_WEIGHTS)
        snapshot = dict(original)
 
        patterns = self._base_patterns(energy_gap=0.8)
        adjust_weights(original, patterns)
 
        assert original == snapshot
 
 
# ══════════════════════════════════════════════
# LAYER 3 — multi-round integration
# ══════════════════════════════════════════════
 
class TestFeedbackLoopIntegration:
 
    @pytest.fixture
    def song_catalog(self):
        """Small catalog with a clear energy split to test learning."""
        return [
            make_song(title="High Energy Pop",    genre="pop",  energy=0.9, valence=0.8, mood="happy"),
            make_song(title="High Energy Rock",   genre="rock", energy=0.85, valence=0.7, mood="intense"),
            make_song(title="Chill Pop",           genre="pop",  energy=0.3, valence=0.6, mood="chill"),
            make_song(title="Chill Jazz",          genre="jazz", energy=0.25, valence=0.5, mood="relaxed"),
            make_song(title="Mid Energy Pop",      genre="pop",  energy=0.5, valence=0.5, mood="happy"),
            make_song(title="Acoustic Ballad",     genre="pop",  energy=0.2, valence=0.4, acousticness=0.9, mood="chill"),
            make_song(title="Dance Floor Hit",     genre="pop",  energy=0.8, valence=0.9, danceability=0.9, mood="happy"),
            make_song(title="Slow Jazz",           genre="jazz", energy=0.15, valence=0.3, mood="relaxed"),
        ]
 
    def test_energy_weight_increases_when_user_prefers_high_energy(self, song_catalog):
        """
        Simulate a user who always likes high-energy songs and dislikes low-energy ones.
        After 3 rounds the energy weight should be higher than where it started.
        """
        user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.5,
                       "danceability": 0.5, "likes_acoustic": False}
        weights = dict(DEFAULT_WEIGHTS)
        initial_energy_weight = weights["energy"]
 
        for _ in range(3):
            # Get recommendations
            recs = recommend_songs(user_prefs, song_catalog, k=5, weights=weights)
 
            # Simulate user: like anything above 0.7 energy, dislike below 0.4
            feedback = []
            for song, score, explanation in recs:
                if song["energy"] >= 0.7:
                    rating = "like"
                elif song["energy"] <= 0.4:
                    rating = "dislike"
                else:
                    rating = "skip"
                feedback.append({"song": song, "rating": rating})
 
            patterns = analyze_feedback(feedback)
            weights, reasoning = adjust_weights(weights, patterns)
 
        assert weights["energy"] > initial_energy_weight, (
            f"Energy weight should have increased from {initial_energy_weight} "
            f"but ended at {weights['energy']}"
        )
 
    def test_genre_weight_decreases_when_user_likes_across_genres(self, song_catalog):
        """
        Simulate a user who likes songs regardless of genre (just cares about energy).
        Genre weight should decrease since it's not the signal.
        """
        user_prefs = {"mood": "happy", "energy": 0.7, "valence": 0.5,
                       "danceability": 0.5, "likes_acoustic": False}
        weights = dict(DEFAULT_WEIGHTS)
        initial_genre_weight = weights["genre"]
 
        for _ in range(3):
            recs = recommend_songs(user_prefs, song_catalog, k=5, weights=weights)
 
            # Simulate user: likes high energy regardless of genre
            feedback = []
            for song, score, explanation in recs:
                if song["energy"] >= 0.6:
                    rating = "like"
                else:
                    rating = "dislike"
                feedback.append({"song": song, "rating": rating})
 
            patterns = analyze_feedback(feedback)
            weights, reasoning = adjust_weights(weights, patterns)
 
        assert weights["genre"] < initial_genre_weight, (
            f"Genre weight should have decreased from {initial_genre_weight} "
            f"but ended at {weights['genre']}"
        )
 
    def test_weights_remain_valid_after_multiple_rounds(self, song_catalog):
        """After several rounds of adjustments, weights should still be well-formed."""
        user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.5,
                       "danceability": 0.5, "likes_acoustic": False}
        weights = dict(DEFAULT_WEIGHTS)
 
        for _ in range(5):
            recs = recommend_songs(user_prefs, song_catalog, k=5, weights=weights)
 
            # Random-ish pattern: like first two, dislike last two, skip middle
            feedback = []
            for i, (song, score, explanation) in enumerate(recs):
                if i < 2:
                    rating = "like"
                elif i >= 3:
                    rating = "dislike"
                else:
                    rating = "skip"
                feedback.append({"song": song, "rating": rating})
 
            patterns = analyze_feedback(feedback)
            weights, _ = adjust_weights(weights, patterns)
 
        # Structural checks
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
        assert all(v >= 0.04 for v in weights.values()), "A weight dropped below floor"
        assert all(v <= 0.41 for v in weights.values()), "A weight exceeded ceiling"
        assert set(weights.keys()) == {"genre", "mood", "energy", "valence", "danceability", "acousticness"}