import numpy as np
from cornac.metrics import RatingMetric
import pandas as pd


class GenreMapEdited(RatingMetric):
    def __init__(self, gender_df, unique_genres, top_k, **kwargs):
        """
        initializating genders of the users
        Parameters
        ----------
        gender_mapping : dict
            A dictionary mapping user IDs to their genders.
        """
        super().__init__(name="GenreMap", **kwargs)
        self.gender_df = gender_df
        self.unique_genres = unique_genres
        self.top_k = top_k

    def compute(self, reco_matrix, item_df):
        """
        reco_matrix : n_userxk np array containing the ranked recommended list for users
        item_df : pd df containing all items with ids and genre info as ohe
        returns the abs diff for each gender genre distribution
        """
        # precision of action = action movies / total action movies
      
        df_reco = pd.DataFrame(
            {
                "userID": np.repeat(np.arange(reco_matrix.shape[0]), self.top_k),
                "itemID": reco_matrix.flatten(),
                "rank": np.tile(np.arange(1, self.top_k + 1), reco_matrix.shape[0]),
            }
        )
        merged_df = pd.merge(df_reco, item_df, on="itemID", how="inner")
        merged_df[self.unique_genres] = merged_df[self.unique_genres].div(
            merged_df[self.unique_genres].sum(axis=1), axis=0
        )
        # merged_df["precision@k"] = merged_df[self.unique_genres]
        cumalitive = merged_df.groupby("userID")[self.unique_genres].cumsum()

        for g in self.unique_genres:
            merged_df[f"{g}_precision@k"] = cumalitive[g] / merged_df["rank"]

        new_df = pd.DataFrame()
        for g in self.unique_genres:
            new_df[g] = merged_df[f"{g}_precision@k"] * merged_df[g]
        new_df["userID"] = merged_df["userID"]
        new_df = new_df.groupby("userID")[self.unique_genres].sum()

        all_distribution = item_df.copy(deep=True)
        all_distribution[self.unique_genres] = all_distribution[self.unique_genres].div(
            all_distribution[self.unique_genres].sum(axis=1), axis=0
        )
        all_distribution = all_distribution[self.unique_genres].sum()
        reco_distribution = new_df[self.unique_genres].div(all_distribution, axis=1)

        g_reco_distribution = self.get_gender_genre_dist(reco_distribution)

        return self.genre_result(g_reco_distribution)

    def get_gender_genre_dist(self, user_reco):
        """
        user_reco : is the recommended genre distibution for all users
        """
        recomen_df = pd.merge(user_reco, self.gender_df, on="userID")
        gender_genre_weights_r = recomen_df.groupby("Gender")[self.unique_genres].mean()

        distribution_gender = gender_genre_weights_r.sort_index()
        return distribution_gender

    def genre_result(self, gender_genre_dist):
        """
        gender_genre_dist : the genre distibution for each genre grouped by gender
        """
        gender_genre_dist = gender_genre_dist.to_numpy()
        return gender_genre_dist[0] - gender_genre_dist[1]
