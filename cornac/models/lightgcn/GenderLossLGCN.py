import pandas as pd
import numpy as np
import torch

#### for light gcn we call a 
class GenderLoss(object):
    def __init__(self, gender, users, genres, recommender, top_k):

        unique_users = torch.unique(users)
        recommendations = [
            recommender.rank(u, k=top_k)[0][:top_k] for u in unique_users
        ]
        # print("::::::")
        # print(recommender.score(1, 3))
        # print(recommender.score(2, 3))
        # print(recommender.score(1, 4))
        # print("::::::")

        reco_df = pd.DataFrame(
            {
                "userID": np.repeat(unique_users, top_k),
                "itemID": np.concatenate(recommendations),
            }
        )

        users = pd.DataFrame(
            {
                "userID": users.numpy(),
                "Gender": gender.numpy(),
            }
        )
        users = users.drop_duplicates()

        self.unique_genres = [
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

        movies = pd.DataFrame(genres, columns=self.unique_genres)

        movies["itemID"] = range(0, len(movies))

        movies[self.unique_genres] = movies[self.unique_genres].div(
            movies[self.unique_genres].sum(axis=1), axis=0
        )
        self.final_df = pd.merge(reco_df, movies, on="itemID", how="inner")

        self.final_df = self.final_df.groupby("userID")[self.unique_genres].mean()

        self.final_df = pd.merge(self.final_df, users, on="userID", how="left")

        gender_genre_weights_r = self.final_df.groupby("Gender")[
            self.unique_genres
        ].mean()

        self.gender_genre_weights = gender_genre_weights_r.sort_index()

    def compute(self):
        uall = torch.tensor(self.gender_genre_weights.values)
        diff = torch.abs(uall[0] - uall[1])
        retVal = torch.sum(diff)

        return retVal
