import pandas as pd


class GenderLossMF:
    def __init__(self, gender, users, items, rating_diff, genres):
        df = pd.DataFrame(
            {
                "userID": users.numpy(),
                "itemID": items.numpy(),
                "rating_error": rating_diff.numpy(),
                "Gender": gender.numpy(),
            }
        )
        unique_genres = [
            "Action",
            "Thriller",
            "Romance",
            "Western",
            "Children's",
            "Mystery",
            "Fantasy",
            "Film-Noir",
            "Documentary",
            "Comedy",
            "Adventure",
            "Sci-Fi",
            "Horror",
            "Crime",
            "Musical",
            "War",
            "Animation",
            "Drama",
        ]
        genre_df = pd.DataFrame(genres.to_numpy(), columns=unique_genres)
        df = pd.concat([df, genre_df], axis=1)
        print(df)
