from recommender import load_songs
from feedback_agent import run_feedback_loop


def main() -> None:
    songs = load_songs("data/songs.csv")

    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "valence": 0.8,
        "danceability": 0.75,
        "likes_acoustic": False,
    }

    run_feedback_loop(user_prefs, songs, k=5, rounds=3)


if __name__ == "__main__":
    main()
