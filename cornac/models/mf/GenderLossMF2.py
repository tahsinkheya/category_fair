import pandas as pd
import numpy as np
import torch


class GenderLossMF2(object):
    def __init__(self, gender, users, items, rating_diff, genres, recommender, top_k):

        # pd_data = []
        # unique_users = np.unique(users.numpy())
        unique_users = torch.unique(users)
        # print(unique_users)
        # print(unique_users.__len__())
        # print(torch.unique(users).__len__())
        # print(torch.unique(users))
        # for i in range(len(unique_users)):
        #     recos = recommender.rank(unique_users[i], k=top_k)[0][:top_k]
        #     for j in range(top_k):
        #         r = {"userID": unique_users[i], "itemID": recos[j]}
        #         pd_data.append(r)
        # reco_df = pd.DataFrame(pd_data)
        # reco_df["itemID"] = reco_df["itemID"].astype(int)

        # ________
        recommendations = [
            recommender.rank(u, k=top_k)[0][:top_k] for u in unique_users
        ]
        reco_df = pd.DataFrame(
            {
                "userID": np.repeat(unique_users, top_k),
                "itemID": np.concatenate(recommendations),
            }
        )
        # print(">>>>>>")
        # print(recommender.score(1, 3))
        # print(">>>>>>")
        # ________

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
        # movies = pd.concat([movies, genre_df], axis=1)

        movies["itemID"] = range(0, len(movies))

        movies[self.unique_genres] = movies[self.unique_genres].div(
            movies[self.unique_genres].sum(axis=1), axis=0
        )

        # self.final_df = pd.merge()

        # for g in self.unique_genres:
        #     self.df[g] = self.df["rating_error"] * self.df[g]
        # df = df.groupby("userID")[self.unique_genres].mean()

        ##########################################
        self.final_df = pd.merge(reco_df, movies, on="itemID", how="inner")

        self.final_df = self.final_df.groupby("userID")[self.unique_genres].mean()

        self.final_df = pd.merge(self.final_df, users, on="userID", how="left")
        # print(movies)
        # print(reco_df)

        gender_genre_weights_r = self.final_df.groupby("Gender")[
            self.unique_genres
        ].mean()
        ##########################################

        self.gender_genre_weights = gender_genre_weights_r.sort_index()

    def compute(self):
        ##########################################

        # uall = self.gender_genre_weights.to_numpy()
        uall = torch.tensor(self.gender_genre_weights.values)

        ##########################################

        # u1 = self.gender_genre_weights_r.iloc[0]
        # u2 = self.gender_genre_weights_r.iloc[1]
        # genre_dist_u1 = u1[self.unique_genres].sum()
        # genre_dist_u2 = u2[self.unique_genres].sum()
        ##########################################

        # diff = np.abs(uall[0] - uall[1])
        diff = torch.abs(uall[0] - uall[1])
        # retVal = np.sum(diff)
        retVal = torch.sum(diff)

        # print(f"diff{diff} retVal{retVal} oldDiff{abs(uall[0]-uall[1])}")
        ##########################################

        return retVal
