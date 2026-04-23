import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

DEFAULT_WEIGHTS = {
    "genre": 0.17,
    "mood": 0.17,
    "energy": 0.17,
    "valence": 0.17,
    "danceability": 0.16,
    "acousticness": 0.16,
}

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        scored = []
        for song in self.songs:
            s = 0.0
            if song.genre == user.favorite_genre:
                s += 1.0
            if song.mood == user.favorite_mood:
                s += 1.0
            s += 1.0 - abs(song.energy - user.target_energy)
            if user.likes_acoustic and song.acousticness > 0.6:
                s += 0.5
            scored.append((s, song))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [song for _, song in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        reasons = []
        if song.genre == user.favorite_genre:
            reasons.append(f"genre:{song.genre}")
        if song.mood == user.favorite_mood:
            reasons.append(f"mood:{song.mood}")
        if abs(song.energy - user.target_energy) < 0.25:
            reasons.append(f"energy:{song.energy:.2f}")
        return ", ".join(reasons) if reasons else "general match"

def load_songs(csv_path: str) -> List[Dict]:
    """Loads songs from a CSV file and casts numeric fields to float."""
    numeric = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for field in numeric:
                row[field] = float(row[field])
            songs.append(row)
    return songs


def score_song(
    user_prefs: Dict, song: Dict, weights: Optional[Dict] = None
) -> Tuple[float, str]:
    """
    Scores a single song against user preferences across all six features.
    Returns (score, explanation). weights defaults to DEFAULT_WEIGHTS.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    score = 0.0
    reasons = []

    if song["genre"] == user_prefs.get("genre"):
        score += w["genre"]
        reasons.append(f"genre:{song['genre']}")

    if song["mood"] == user_prefs.get("mood"):
        score += w["mood"]
        reasons.append(f"mood:{song['mood']}")

    energy_sim = 1.0 - abs(song["energy"] - user_prefs.get("energy", 0.5))
    score += w["energy"] * energy_sim
    if energy_sim > 0.75:
        reasons.append(f"energy:{song['energy']:.2f}")

    valence_sim = 1.0 - abs(song["valence"] - user_prefs.get("valence", 0.5))
    score += w["valence"] * valence_sim
    if valence_sim > 0.75:
        reasons.append(f"valence:{song['valence']:.2f}")

    dance_sim = 1.0 - abs(song["danceability"] - user_prefs.get("danceability", 0.5))
    score += w["danceability"] * dance_sim
    if dance_sim > 0.75:
        reasons.append(f"danceability:{song['danceability']:.2f}")

    acoust_val = (
        song["acousticness"]
        if user_prefs.get("likes_acoustic", False)
        else (1.0 - song["acousticness"])
    )
    score += w["acousticness"] * acoust_val
    if acoust_val > 0.75:
        reasons.append(f"acousticness:{song['acousticness']:.2f}")

    return score, (", ".join(reasons) if reasons else "general match")


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    weights: Optional[Dict] = None,
) -> List[Tuple[Dict, float, str]]:
    """Score all songs and return top-k as (song, score, explanation)."""
    scored = [(song, *score_song(user_prefs, song, weights)) for song in songs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
