import pandas as pd


class GenderLossMF(object):
    def __init__(self, gender, users, items, rating_diff, genres):
        self.df = pd.DataFrame(
            {
                "userID": users.numpy(),
                "itemID": items.numpy(),
                "rating_error": rating_diff.detach().numpy(),
                "Gender": gender.numpy(),
            }
        )
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
        genre_df = pd.DataFrame(genres.numpy(), columns=self.unique_genres)
        self.df = pd.concat([self.df, genre_df], axis=1)
        self.df[self.unique_genres] = self.df[self.unique_genres].div(self.df[self.unique_genres].sum(axis=1), axis=0)
        
        
        for g in self.unique_genres:
            self.df[g] = self.df["rating_error"] * self.df[g]
        # df = df.groupby("userID")[self.unique_genres].mean()
        self.gender_genre_weights_r = self.df.groupby("Gender")[self.unique_genres].mean()

    def compute(self):
        u1 = self.gender_genre_weights_r.iloc[0]
        u2 = self.gender_genre_weights_r.iloc[1]
        genre_dist_u1 = u1[self.unique_genres].sum()
        genre_dist_u2 = u2[self.unique_genres].sum()
      
        
        return abs(genre_dist_u1 - genre_dist_u2)
