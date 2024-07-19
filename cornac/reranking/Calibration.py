# tahsin kheya
# last modified 21/05/2024
import pandas as pd
import os
import numpy as np
import torch
import time
from logging import getLogger


class Calibration(object):
    def __init__(self, config, movies):
        # self.logger = getLogger()
        self.reco_distribution = []
        self.kl = []
        # self.logger = getLogger()

        self.gkl = []
        self.genders = []
        self.actual_genre_dist = pd.read_csv(
            os.path.join(config["user_genre_dist_file"]),
            sep="\t",
        )
        self.config = config
        self.item_df = movies
        self.actual_distribution = self.actual_genre_dist.to_numpy()
        self.actual_distribution_gender = []

    def get_actual_genre_dist(self):
        pass

    def calculate_kl_divergence(actual_dist, reco_dist, a=0.01):
        retVal = 0
        pass

    def get_possible_items(scores, pos_items):
        """for a given user, return items that can be recommended to them in the future. this list wont
        have items user has already interacted with"""
        pass

    def get_all_user_recommended_genre_dist(self, topk_reco):

        df_reco = pd.DataFrame(
            {
                "userID": np.repeat(
                    np.arange(topk_reco.shape[0]), max(self.config["topk"])
                ),
                "itemID": topk_reco.flatten(),
                "rank": np.tile(
                    np.arange(1, max(self.config["topk"]) + 1), topk_reco.shape[0]
                ),
            }
        )
        df_reco["weight_factor"] = 1 / (df_reco["rank"]) ** 0.1
        merged_df = pd.merge(df_reco, self.item_df, on="itemID", how="inner")
        merged_df[self.config["genre_columns"]] = merged_df[
            self.config["genre_columns"]
        ].div(merged_df[self.config["genre_columns"]].sum(axis=1), axis=0)
        merged_df[self.config["genre_columns"]] = (
            merged_df["weight_factor"].values[:, None]
            * merged_df[self.config["genre_columns"]]
        )
        reco_distribution = merged_df[["userID"] + self.config["genre_columns"]]
        reco_distribution = reco_distribution.groupby("userID")[
            self.config["genre_columns"]
        ].mean()
        reco_distribution[self.config["genre_columns"]] = reco_distribution[
            self.config["genre_columns"]
        ].div(reco_distribution[self.config["genre_columns"]].sum(axis=1), axis=0)

        return reco_distribution

    def get_recom_distribution(self, reco):
        reco = np.array(reco)
        weights = 1 / (np.arange(len(reco)) + 1) ** 0.1
        item_genre_weights = np.zeros((len(reco), len(self.config["genre_columns"])))
        for i, itemID in enumerate(reco):
            item_index = self.item_df.index.get_loc(itemID)
            item_genre_weights[i] = (
                weights[i] * self.item_df.loc[item_index, self.config["genre_columns"]]
            )
        # Normalize genre weights so sum is 1
        item_genre_weights /= np.sum(item_genre_weights, axis=1, keepdims=True)
        return item_genre_weights

    def get_kl_div(self, q_dist, p_dist, a, uid):
        kl_div = 0
        qg_u = (1 - a) * q_dist + a * p_dist
        nonzero_indices = np.where(p_dist != 0)
        kl_div = np.sum(
            p_dist[nonzero_indices]
            * np.log10(p_dist[nonzero_indices] / qg_u[nonzero_indices])
        )

        return kl_div

    def get_gender_representation_one_user(self, reco_dist, uid):
        c = []
        retVal = []
        print(".......")
        print(reco_dist)
        print(self.actual_distribution[uid])

        for i in range(len(self.config["genre_columns"])):
            c.append(np.log(reco_dist[i] / self.actual_distribution[uid][i + 1]))
        # Assuming self.config["genre_columns"] is a list or array-like
        genre_columns_count = len(self.config["genre_columns"])

        # Slice the necessary parts of reco_dist and actual_distribution
        reco_dist_slice = reco_dist[:genre_columns_count]
        actual_dist_slice = self.actual_distribution[uid][1 : genre_columns_count + 1]

        # Compute the logarithms of the ratio
        log_ratios = np.log(reco_dist_slice / actual_dist_slice)

        # Convert log_ratios to a list and append to retVal
        retVal.extend(log_ratios.tolist())

    def compute_diversity_score(self, reco_items, uid, l, scores, user_reco, b):

        # -----------------the diversity term----------------------------------
        alpha = 0.01
        # reco_items=self.reco_distribution[uid]
        sum_score = 0
        reco_dist = self.get_recom_distribution(reco_items)

        sum_dist = np.sum(reco_dist, axis=0)
        avg_dist = sum_dist / reco_dist.shape[0]
        # 1x18 array (it already is, but let's reshape explicitly)
        reco_dist = avg_dist.reshape(1, -1)
        # normalise values
        reco_dist = reco_dist / sum(reco_dist[0])

        kl = self.get_kl_div(
            reco_dist[0], self.actual_distribution[uid][1:], alpha, uid
        )
        # -----------------the diversity term----------------------------------

        # -----------------the fairness term----------------------------------
        # recommended dist mean for each gender
        male_user_ids = [
            user_id for user_id, gender in self.genders.items() if gender == 0
        ]
        gender_genre_dist = self.actual_distribution_gender
        # gender_reco_dist = self.get_gender_representation_one_user(reco_dist[0],uid)
        if uid in male_user_ids:
            compare_dist = gender_genre_dist.to_numpy()[1]
        else:
            compare_dist = gender_genre_dist.to_numpy()[0]

        gender_kl = self.get_kl_div(reco_dist[0], compare_dist, alpha, uid)

        # -----------------the fairness term----------------------------------

        for r in range(len(reco_items)):
            sum_score += scores[uid][reco_items[r]]

        # if len(reco_items) == 5:
        #     self.kl.append(kl)
        #     self.gkl.append(gender_kl)
        #     print(":;;;;;;;;;;;")
        #     print(uid)

        #     print(self.kl)
        #     print(self.gkl)
        #     print(":;;;;;;;;;;;")

        return (1 - l - b) * sum_score - l * kl - b * gender_kl

    def get_improved_reco(self, items, scores, users):
        self.genders = genders
        # reco = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        # scores = torch.cat(batch_score_list, dim=0).cpu().numpy()
        # self.reco_distribution = self.get_recom_distribution(reco)
        return self.get_new_recommendations(reco=items, scores=scores)

    def get_kl_div_gender(self, female_dist, male_dist, a):
        kl_div = 0
        for i in range(len(self.config["genre_columns"])):
            female_dist = (1 - a) * female_dist[i] + a * male_dist[i]
            if male_dist[i] == 0:
                continue
            kl_div = kl_div + male_dist[i] * np.log10(male_dist[i] / female_dist[i])

        return kl_div

    def get_gender_genre_dist(self, user_reco):
        gender_df = pd.DataFrame(self.genders.items(), columns=["userID", "gender"])
        actual_dist_gender = pd.merge(self.actual_genre_dist, gender_df, on="userID")
        recomen_df = pd.merge(user_reco, gender_df, on="userID")
        gender_genre_weights_r = recomen_df.groupby("gender")[
            self.config["genre_columns"]
        ].mean()
        gender_genre_weights_a = actual_dist_gender.groupby("gender")[
            self.config["genre_columns"]
        ].mean()
        self.actual_distribution_gender = gender_genre_weights_a.sort_index()

    def get_new_recommendations(self, reco, scores):
        """reco is 6040x50 and scores is 6040x3416"""

        user_reco_dist = self.get_all_user_recommended_genre_dist(reco)
        self.get_gender_genre_dist(user_reco_dist)

        # gender_discriminated_agaisnt = self.get_gender_discriminated_agaisnt(
        #     user_reco_dist
        # )
        # male_user_ids = [user_id for user_id, gender in self.genders.items() if gender == 0]
        # female_user_ids = [user_id for user_id, gender in self.genders.items() if gender == 1]
        # for each gender clculate the skew for each genre:
        l = 0.69  # lambda
        b = 0.29  # beta
        all_users = []
        # male_active_users= [4168,1679,4276]
        male_active_users = [4168]
        female_active_users = [1149]
        # female_active_users= [1149,1087,3223]
        male_inactive_users = [1712]
        # male_inactive_users= [1712,4067,2713]
        female_inactive_users = [2583]
        # female_inactive_users= [2583,2159,97]
        test_users = (
            male_active_users
            + female_active_users
            + male_inactive_users
            + female_inactive_users
        )
        top_k = max(self.config["topk"])

        n_users, n_items = scores.shape

        for u in range(n_users):
            remaining_items = list(range(n_items))
            u_calibrated = []
            for k in range(top_k):
                diversity_scores = [
                    self.compute_diversity_score(
                        u_calibrated + [i], u, l, scores, user_reco_dist, b
                    )
                    for i in remaining_items
                ]

                max_index = np.argmax(diversity_scores)
                best_item = remaining_items[max_index]
                u_calibrated.append(best_item)
                remaining_items.pop(max_index)
                self.logger.info(u_calibrated)

            self.logger.info(u_calibrated)

            all_users.append(u_calibrated)

        return np.array(all_users)
