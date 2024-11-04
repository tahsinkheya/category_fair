# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from collections import OrderedDict
import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm
import random
from ..data import FeatureModality
from ..data import TextModality, ReviewModality
from ..data import ImageModality
from ..data import GraphModality
from ..data import SentimentModality
from ..data import Dataset
from ..metrics import RatingMetric
from ..metrics import RankingMetric
from ..experiment.result import Result
from ..utils import get_rng


def rating_eval(model, metrics, test_set, user_based=False, verbose=False):
    """Evaluate model on provided rating metrics.

    Parameters
    ----------
    model: :obj:`cornac.models.Recommender`, required
        Recommender model to be evaluated.

    metrics: :obj:`iterable`, required
        List of rating metrics :obj:`cornac.metrics.RatingMetric`.

    test_set: :obj:`cornac.data.Dataset`, required
        Dataset to be used for evaluation.

    user_based: bool, optional, default: False
        Evaluation mode. Whether results are averaging based on number of users or number of ratings.

    verbose: bool, optional, default: False
        Output evaluation progress.

    Returns
    -------
    res: (List, List)
        Tuple of two lists:
         - average result for each of the metrics
         - average result per user for each of the metrics

    """

    if len(metrics) == 0:
        return [], []

    avg_results = []
    user_results = []

    (u_indices, i_indices, r_values) = test_set.uir_tuple
    r_preds = np.fromiter(
        tqdm(
            (
                model.rate(user_idx, item_idx).item()
                for user_idx, item_idx in zip(u_indices, i_indices)
            ),
            desc="Rating",
            disable=not verbose,
            miniters=100,
            total=len(u_indices),
        ),
        dtype="float",
    )

    gt_mat = test_set.csr_matrix
    pd_mat = csr_matrix((r_preds, (u_indices, i_indices)), shape=gt_mat.shape)

    #############
    # implementation of beyond parity U_val metric
    unique_i_indices = np.unique(i_indices)
    genders = model.user_features

    # U_val = 0
    # for item in unique_i_indices:
    #     currnet_item_ind = i_indices == item
    #     u_i_ind = u_indices[currnet_item_ind]
    #     r_gt_i_ind = r_values[currnet_item_ind]
    #     r_pred_i_ind = r_preds[currnet_item_ind]
    #     item_genders = genders[u_i_ind]
    #     u_male = item_genders == 0
    #     u_female = item_genders == 1
    #     # female avg pred score
    #     E_g_yj = r_pred_i_ind[u_female].mean() if u_female.any() else 0
    #     # male avg pred score
    #     E_mg_yj = r_pred_i_ind[u_male].mean() if u_male.any() else 0
    #     E_g_rj = r_gt_i_ind[u_female].mean() if u_female.any() else 0
    #     E_mg_rj = r_gt_i_ind[u_male].mean() if u_male.any() else 0
    #     U_val = U_val + abs((E_g_yj - E_mg_yj) - (E_g_rj - E_mg_rj))

    # print("::::::")
    # print(U_val / len(unique_i_indices))

    #     ##$$$$$$$$
    #     import numpy as np

    # # Get unique item indices
    # # unique_iids = np.unique(i_indices)

    # # Initialize a list to store results
    # # abs_diffs = []

    # # Loop through each unique item id (iid)
    # for iid in unique_iids:
    #     # Get the indices where the item id matches this iid
    #     item_mask = (i_indices == iid)

    #     # Filter the users and their corresponding ratings (ground truth and predicted)
    #     item_u_indices = u_indices[item_mask]    # User indices for this item
    #     item_r_values = r_values[item_mask]      # Ground truth ratings for this item
    #     item_r_preds = r_preds[item_mask]        # Predicted ratings for this item
    #     item_genders = genders[item_mask]
    #     # Genders of the users for this item

    #     # Calculate the rating differences (r_values - r_preds) for this item
    #     rating_diffs = item_r_values - item_r_preds

    #     # Separate the differences by gender
    #     male_mask = (item_genders == 1)
    #     female_mask = (item_genders == 0)

    #     male_diffs = rating_diffs[male_mask]     # Rating differences for male users
    #     female_diffs = rating_diffs[female_mask] # Rating differences for female users

    #     # Compute the average difference for male and female users (if they exist)
    #     avg_male_diff = np.mean(male_diffs) if len(male_diffs) > 0 else np.nan
    #     avg_female_diff = np.mean(female_diffs) if len(female_diffs) > 0 else np.nan

    #     # Calculate the absolute difference between male and female average differences
    #     if not np.isnan(avg_male_diff) and not np.isnan(avg_female_diff):
    #         abs_diff = abs(avg_male_diff - avg_female_diff)
    #         abs_diffs.append((iid, abs_diff))  # Store item id and absolute difference

    # # Print or analyze the resulting absolute differences
    # for iid, abs_diff in abs_diffs:
    #     print(f"Item ID: {iid}, Absolute Difference: {abs_diff}")

    #############

    test_user_indices = set(u_indices)
    for mt in metrics:

        if user_based:  # averaging over users
            user_results.append(
                {
                    user_idx: mt.compute(
                        gt_ratings=gt_mat.getrow(user_idx).data,
                        pd_ratings=pd_mat.getrow(user_idx).data,
                    ).item()
                    for user_idx in test_user_indices
                }
            )
            avg_results.append(sum(user_results[-1].values()) / len(user_results[-1]))
        else:  # averaging over ratings
            user_results.append({})
            avg_results.append(mt.compute(gt_ratings=r_values, pd_ratings=r_preds))

    return avg_results, user_results


def ranking_eval(
    model,
    metrics,
    train_set,
    test_set,
    val_set=None,
    rating_threshold=1.0,
    exclude_unknowns=True,
    verbose=False,
):
    """Evaluate model on provided ranking metrics.

    Parameters
    ----------
    model: :obj:`cornac.models.Recommender`, required
        Recommender model to be evaluated.

    metrics: :obj:`iterable`, required
        List of rating metrics :obj:`cornac.metrics.RankingMetric`.

    train_set: :obj:`cornac.data.Dataset`, required
        Dataset to be used for model training. This will be used to exclude
        observations already appeared during training.

    test_set: :obj:`cornac.data.Dataset`, required
        Dataset to be used for evaluation.

    val_set: :obj:`cornac.data.Dataset`, optional, default: None
        Dataset to be used for model selection. This will be used to exclude
        observations already appeared during validation.

    rating_threshold: float, optional, default: 1.0
        The threshold to convert ratings into positive or negative feedback.

    exclude_unknowns: bool, optional, default: True
        Ignore unknown users and items during evaluation.

    verbose: bool, optional, default: False
        Output evaluation progress.

    Returns
    -------
    res: (List, List)
        Tuple of two lists:
         - average result for each of the metrics
         - average result per user for each of the metrics

    """

    if len(metrics) == 0:
        return [], []

    max_k = max(m.k for m in metrics)

    avg_results = []
    user_results = [{} for _ in enumerate(metrics)]

    test_mat = test_set.csr_matrix
    train_mat = train_set.csr_matrix
    val_mat = None if val_set is None else val_set.csr_matrix

    def pos_items(csr_row):
        return [
            item_idx
            for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
            if rating >= rating_threshold
        ]

    test_user_indices = set(test_set.uir_tuple[0])

    for user_idx in tqdm(
        test_user_indices, desc="Ranking", disable=not verbose, miniters=100
    ):
        test_pos_items = pos_items(test_mat.getrow(user_idx))

        if len(test_pos_items) == 0:
            continue

        # binary mask for ground-truth positive items
        u_gt_pos_mask = np.zeros(test_set.num_items, dtype="int")
        u_gt_pos_mask[test_pos_items] = 1

        val_pos_items = [] if val_mat is None else pos_items(val_mat.getrow(user_idx))
        train_pos_items = (
            pos_items(train_mat.getrow(user_idx))
            if user_idx < train_mat.shape[0]
            else []
        )

        # binary mask for ground-truth negative items, removing all positive items

        u_gt_neg_mask = np.ones(test_set.num_items, dtype="int")
        u_gt_neg_mask[test_pos_items + val_pos_items + train_pos_items] = 0

        # filter items being considered for evaluation
        if exclude_unknowns:
            u_gt_pos_mask = u_gt_pos_mask[: train_set.num_items]
            u_gt_neg_mask = u_gt_neg_mask[: train_set.num_items]

        # so this is bascially the items that are rated above threshold and want interacted with
        item_indices = np.nonzero(u_gt_pos_mask + u_gt_neg_mask)[0]
        u_gt_pos_items = np.nonzero(u_gt_pos_mask)[0]
        u_gt_neg_items = np.nonzero(u_gt_neg_mask)[0]

        item_rank, item_scores = model.rank(
            user_idx=user_idx, item_indices=item_indices, k=max_k
        )

        for i, mt in enumerate(metrics):
            mt_score = mt.compute(
                gt_pos=u_gt_pos_items,
                gt_neg=u_gt_neg_items,
                pd_rank=item_rank,
                pd_scores=item_scores,
                item_indices=item_indices,
            )
            user_results[i][user_idx] = mt_score

            # if mt.name == "Recall@50" :
            #     for userid in test_user_indices:
            #         print("......")
            #         print(userid)
            #         print(model.user_features[userid])
            #         print(model.user_features)
            #         print("......")

    # avg results of ranking metrics
    for i, mt in enumerate(metrics):
        avg_results.append(sum(user_results[i].values()) / len(user_results[i]))

    return avg_results, user_results


class BaseMethod:
    """Base Evaluation Method

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the triplet format [(user_id, item_id, rating_value)].

    fmt: str, default: 'UIR'
        Format of the input data. Currently, we are supporting:

        'UIR': User, Item, Rating
        'UIRT': User, Item, Rating, Timestamp

    rating_threshold: float, optional, default: 1.0
        Threshold used to binarize rating values into positive or negative feedback for
        model evaluation using ranking metrics (rating metrics are not affected).

    seed: int, optional, default: None
        Random seed for reproducibility.

    exclude_unknowns: bool, optional, default: True
        If `True`, unknown users and items will be ignored during model evaluation.

    verbose: bool, optional, default: False
        Output running log.

    """

    def __init__(
        self,
        data=None,
        fmt="UIR",
        rating_threshold=1.0,
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        **kwargs,
    ):
        self.data = data
        self.fmt = fmt
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.rating_threshold = rating_threshold
        self.exclude_unknowns = exclude_unknowns
        self.verbose = verbose
        self.seed = seed
        self.rng = get_rng(seed)
        self.global_uid_map = kwargs.get("global_uid_map", OrderedDict())
        self.global_iid_map = kwargs.get("global_iid_map", OrderedDict())
        self.global_uid_gender_map = kwargs.get("global_uid_gender_map", OrderedDict())
        self.global_iid_cat_map = kwargs.get("global_iid_cat_map", OrderedDict())

        self.user_feature = kwargs.get("user_feature", None)
        self.user_text = kwargs.get("user_text", None)
        self.user_image = kwargs.get("user_image", None)
        self.user_graph = kwargs.get("user_graph", None)
        self.item_feature = kwargs.get("item_feature", None)
        self.item_text = kwargs.get("item_text", None)
        self.item_image = kwargs.get("item_image", None)
        self.item_graph = kwargs.get("item_graph", None)
        self.sentiment = kwargs.get("sentiment", None)
        self.review_text = kwargs.get("review_text", None)
        random.seed(seed)

        if verbose:
            print("rating_threshold = {:.1f}".format(rating_threshold))
            print("exclude_unknowns = {}".format(exclude_unknowns))

    @property
    def total_users(self):
        return len(self.global_uid_map)

    @property
    def total_items(self):
        return len(self.global_iid_map)

    @property
    def user_feature(self):
        return self.__user_feature

    @property
    def user_text(self):
        return self.__user_text

    @user_feature.setter
    def user_feature(self, input_modality):
        if input_modality is not None and not isinstance(
            input_modality, FeatureModality
        ):
            raise ValueError(
                "input_modality has to be instance of FeatureModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_feature = input_modality

    @user_text.setter
    def user_text(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, TextModality):
            raise ValueError(
                "input_modality has to be instance of TextModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_text = input_modality

    @property
    def user_image(self):
        return self.__user_image

    @user_image.setter
    def user_image(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, ImageModality):
            raise ValueError(
                "input_modality has to be instance of ImageModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_image = input_modality

    @property
    def user_graph(self):
        return self.__user_graph

    @user_graph.setter
    def user_graph(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, GraphModality):
            raise ValueError(
                "input_modality has to be instance of GraphModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_graph = input_modality

    @property
    def item_feature(self):
        return self.__item_feature

    @property
    def item_text(self):
        return self.__item_text

    @item_feature.setter
    def item_feature(self, input_modality):
        if input_modality is not None and not isinstance(
            input_modality, FeatureModality
        ):
            raise ValueError(
                "input_modality has to be instance of FeatureModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_feature = input_modality

    @item_text.setter
    def item_text(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, TextModality):
            raise ValueError(
                "input_modality has to be instance of TextModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_text = input_modality

    @property
    def item_image(self):
        return self.__item_image

    @item_image.setter
    def item_image(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, ImageModality):
            raise ValueError(
                "input_modality has to be instance of ImageModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_image = input_modality

    @property
    def item_graph(self):
        return self.__item_graph

    @item_graph.setter
    def item_graph(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, GraphModality):
            raise ValueError(
                "input_modality has to be instance of GraphModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_graph = input_modality

    @property
    def sentiment(self):
        return self.__sentiment

    @sentiment.setter
    def sentiment(self, input_modality):
        if input_modality is not None and not isinstance(
            input_modality, SentimentModality
        ):
            raise ValueError(
                "input_modality has to be instance of SentimentModality but {}".format(
                    type(input_modality)
                )
            )
        self.__sentiment = input_modality

    @property
    def review_text(self):
        return self.__review_text

    @review_text.setter
    def review_text(self, input_modality):
        if input_modality is not None and not isinstance(
            input_modality, ReviewModality
        ):
            raise ValueError(
                "input_modality has to be instance of ReviewModality but {}".format(
                    type(input_modality)
                )
            )
        self.__review_text = input_modality

    def _reset(self):
        """Reset the random number generator for reproducibility"""
        self.rng = get_rng(self.seed)
        self.test_set = self.test_set.reset()

    @staticmethod
    def organize_metrics(metrics):
        """Organize metrics according to their types (rating or raking)

        Parameters
        ----------
        metrics: :obj:`iterable`
            List of metrics.

        """
        if isinstance(metrics, dict):
            rating_metrics = metrics.get("rating", [])
            ranking_metrics = metrics.get("ranking", [])
        elif isinstance(metrics, list):
            rating_metrics = []
            ranking_metrics = []
            for mt in metrics:
                if isinstance(mt, RatingMetric):
                    rating_metrics.append(mt)
                elif isinstance(mt, RankingMetric) and hasattr(mt.k, "__len__"):
                    ranking_metrics.extend(
                        [mt.__class__(k=_k) for _k in sorted(set(mt.k))]
                    )
                else:
                    ranking_metrics.append(mt)
        else:
            raise ValueError("Type of metrics has to be either dict or list!")

        # sort metrics by name
        rating_metrics = sorted(rating_metrics, key=lambda mt: mt.name)
        ranking_metrics = sorted(ranking_metrics, key=lambda mt: mt.name)
        return rating_metrics, ranking_metrics

    def _build_datasets(
        self,
        train_data,
        test_data,
        val_data=None,
        user_features=None,
        item_features=None,
    ):

        self.train_set = Dataset.build(
            data=train_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            user_features=user_features,
            global_iid_map=self.global_iid_map,
            global_uid_gender_map=self.global_uid_gender_map,
            seed=self.seed,
            item_features=item_features,
            global_iid_cat_map=self.global_iid_cat_map,
            exclude_unknowns=False,
        )

        if self.verbose:
            print("---")
            print("Training data:")
            print(self.train_set.uir_tuple)
            # x = pd.DataFrame(self.train_set.uir_tuple).transpose()
            # x.columns = ["uid", "iid", "rating"]
            # x = x.astype({"uid": "int", "iid": "int", "rating": "int"})
            # r_global_uid_map = {v: k for k, v in self.global_uid_map.items()}
            # r_global_iid_map = {v: k for k, v in self.global_iid_map.items()}

            # x["uid"] = x["uid"].map(r_global_uid_map)
            # x["iid"] = x["iid"].map(r_global_iid_map)

            # x.to_csv("training_set_seed123_ml100k.csv", index=False, header=False)
            print("Number of users = {}".format(self.train_set.num_users))
            print("Number of items = {}".format(self.train_set.num_items))
            print("Number of ratings = {}".format(self.train_set.num_ratings))
            print("Max rating = {:.1f}".format(self.train_set.max_rating))
            print("Min rating = {:.1f}".format(self.train_set.min_rating))
            print("Global mean = {:.1f}".format(self.train_set.global_mean))
            print(
                "Global mean Imolicit= {:.1f}".format(
                    self.train_set.global_mean_implicit
                )
            )

        self.test_set = Dataset.build(
            data=test_data,
            user_features=user_features,
            fmt=self.fmt,
            global_uid_gender_map=self.global_uid_gender_map,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            item_features=item_features,
            global_iid_cat_map=self.global_iid_cat_map,
            seed=self.seed,
            exclude_unknowns=self.exclude_unknowns,
        )
        if self.verbose:
            print("---")
            print("Test data:")
            print("Number of users = {}".format(len(self.test_set.uid_map)))
            print("Number of items = {}".format(len(self.test_set.iid_map)))
            print("Number of ratings = {}".format(self.test_set.num_ratings))
            print(
                "Number of unknown users = {}".format(
                    self.test_set.num_users - self.train_set.num_users
                )
            )
            print(
                "Number of unknown items = {}".format(
                    self.test_set.num_items - self.train_set.num_items
                )
            )
            # x = pd.DataFrame(self.test_set.uir_tuple).transpose()
            # x.columns = ["uid", "iid", "rating"]
            # x = x.astype({"uid": "int", "iid": "int", "rating": "int"})
            # r_global_uid_map = {v: k for k, v in self.global_uid_map.items()}
            # r_global_iid_map = {v: k for k, v in self.global_iid_map.items()}

            # x["uid"] = x["uid"].map(r_global_uid_map)
            # x["iid"] = x["iid"].map(r_global_iid_map)

            # x.to_csv("testing_set_seed123_ml1m.csv", index=False, header=False)

        if val_data is not None and len(val_data) > 0:
            self.val_set = Dataset.build(
                data=val_data,
                fmt=self.fmt,
                global_uid_gender_map=self.global_uid_gender_map,
                global_uid_map=self.global_uid_map,
                global_iid_map=self.global_iid_map,
                seed=self.seed,
                global_iid_cat_map=self.global_iid_cat_map,
                item_features=item_features,
                user_features=user_features,
                exclude_unknowns=self.exclude_unknowns,
            )
            if self.verbose:
                print("---")
                print("Validation data:")
                print("Number of users = {}".format(len(self.val_set.uid_map)))
                print("Number of items = {}".format(len(self.val_set.iid_map)))
                print("Number of ratings = {}".format(self.val_set.num_ratings))
            # x = pd.DataFrame(self.val_set.uir_tuple).transpose()
            # x.columns = ["uid", "iid", "rating"]
            # x = x.astype({"uid": "int", "iid": "int", "rating": "int"})
            # r_global_uid_map = {v: k for k, v in self.global_uid_map.items()}
            # r_global_iid_map = {v: k for k, v in self.global_iid_map.items()}

            # x["uid"] = x["uid"].map(r_global_uid_map)
            # x["iid"] = x["iid"].map(r_global_iid_map)

            # x.to_csv("val_set_seed123_ml100k.csv", index=False, header=False)

        if self.verbose:
            print("---")
            print("Total users = {}".format(self.total_users))
            print("Total items = {}".format(self.total_items))

    def _build_modalities(self):
        for user_modality in [
            self.user_feature,
            self.user_text,
            self.user_image,
            self.user_graph,
        ]:
            if user_modality is None:
                continue
            user_modality.build(
                id_map=self.global_uid_map,
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        for item_modality in [
            self.item_feature,
            self.item_text,
            self.item_image,
            self.item_graph,
        ]:
            if item_modality is None:
                continue
            item_modality.build(
                id_map=self.global_iid_map,
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        for modality in [self.sentiment, self.review_text]:
            if modality is None:
                continue
            modality.build(
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        self.add_modalities(
            user_feature=self.user_feature,
            user_text=self.user_text,
            user_image=self.user_image,
            user_graph=self.user_graph,
            item_feature=self.item_feature,
            item_text=self.item_text,
            item_image=self.item_image,
            item_graph=self.item_graph,
            sentiment=self.sentiment,
            review_text=self.review_text,
        )

    def add_modalities(self, **kwargs):
        """
        Add successfully built modalities to all datasets. This is handy for
        seperately built modalities that are not invoked in the build method.
        """
        self.user_feature = kwargs.get("user_feature", None)
        self.user_text = kwargs.get("user_text", None)
        self.user_image = kwargs.get("user_image", None)
        self.user_graph = kwargs.get("user_graph", None)
        self.item_feature = kwargs.get("item_feature", None)
        self.item_text = kwargs.get("item_text", None)
        self.item_image = kwargs.get("item_image", None)
        self.item_graph = kwargs.get("item_graph", None)
        self.sentiment = kwargs.get("sentiment", None)
        self.review_text = kwargs.get("review_text", None)

        for data_set in [self.train_set, self.test_set, self.val_set]:
            if data_set is None:
                continue
            data_set.add_modalities(
                user_feature=self.user_feature,
                user_text=self.user_text,
                user_image=self.user_image,
                user_graph=self.user_graph,
                item_feature=self.item_feature,
                item_text=self.item_text,
                item_image=self.item_image,
                item_graph=self.item_graph,
                sentiment=self.sentiment,
                review_text=self.review_text,
            )

    def build(
        self,
        train_data,
        test_data,
        val_data=None,
        user_features=None,
        item_features=None,
    ):
        if train_data is None or len(train_data) == 0:
            raise ValueError("train_data is required but None or empty!")
        if test_data is None or len(test_data) == 0:
            raise ValueError("test_data is required but None or empty!")

        self.global_uid_map.clear()
        self.global_iid_map.clear()
        self.global_uid_gender_map.clear()
        self.global_iid_cat_map.clear()

        self._build_datasets(
            train_data, test_data, val_data, user_features, item_features
        )
        self._build_modalities()

        return self

    @staticmethod
    def eval(
        model,
        train_set,
        test_set,
        val_set,
        rating_threshold,
        exclude_unknowns,
        user_based,
        rating_metrics,
        ranking_metrics,
        verbose,
    ):
        """Running evaluation for rating and ranking metrics respectively."""
        metric_avg_results = OrderedDict()
        metric_user_results = OrderedDict()

        avg_results, user_results = rating_eval(
            model=model,
            metrics=rating_metrics,
            test_set=test_set,
            user_based=user_based,
            verbose=verbose,
        )
        for i, mt in enumerate(rating_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

        avg_results, user_results = ranking_eval(
            model=model,
            metrics=ranking_metrics,
            train_set=train_set,
            test_set=test_set,
            val_set=val_set,
            rating_threshold=rating_threshold,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
        )
        for i, mt in enumerate(ranking_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]
        # print(",,,,,,,,")
        genders = model.user_features

        male_active = [
            452,
            7,
            641,
            296,
            797,
            690,
            46,
            474,
            548,
            638,
            859,
            719,
            688,
            75,
            741,
            167,
            706,
            554,
            214,
            559,
            370,
            384,
            678,
            633,
            414,
            484,
            22,
            275,
            934,
            898,
            269,
            52,
            697,
            385,
            535,
            418,
            472,
            48,
            864,
            894,
            203,
            567,
            300,
            659,
            866,
            469,
            11,
            154,
            575,
            732,
            283,
            87,
            689,
            204,
            603,
            200,
            226,
            636,
            168,
            463,
            261,
            490,
            394,
            9,
            25,
            32,
            909,
            29,
            97,
            785,
            225,
            122,
            792,
            766,
            422,
            380,
            19,
            352,
            101,
            833,
            114,
            412,
            171,
            460,
            408,
            578,
            517,
            69,
            662,
            30,
            483,
            545,
            822,
            723,
            910,
            647,
            51,
            102,
            89,
            448,
        ]
        # male_active = male_active[:100]

        female_active = [
            618,
            682,
            470,
            127,
            322,
            930,
            882,
            620,
            303,
            476,
            377,
            365,
            354,
            338,
            381,
            911,
            694,
            179,
            50,
            817,
            264,
            146,
            243,
            595,
            770,
            105,
            96,
            318,
            856,
            488,
            700,
            465,
            929,
            27,
            23,
            895,
            594,
            398,
            362,
            916,
            562,
            655,
            74,
            449,
            151,
            677,
            2,
            336,
            259,
            76,
            66,
            791,
            6,
            113,
            696,
            123,
            832,
            561,
            383,
            693,
            701,
            138,
            369,
            421,
            544,
            400,
            161,
            896,
            103,
            880,
            649,
            771,
            262,
            166,
            374,
            928,
            644,
            491,
            514,
            728,
            768,
            109,
            245,
            15,
            621,
            858,
            543,
            402,
            73,
            308,
            900,
            158,
            287,
            627,
            794,
            533,
            937,
            903,
            219,
            871,
        ]
        # female_active = female_active[:80]
        # male_iids = [62, 256,371, 114, 579, 368, 143, 47, 538, 769, 807, 460, 473, 942, 207, 180, 469, 806, 461, 339, 212, 1, 616, 940, 861, 534, 90, 8, 439, 638, 134, 53, 120, 187, 170, 25, 408, 609, 824, 681, 367, 669, 44, 426, 477, 749, 689, 275, 755, 813, 438, 14, 566, 737, 614, 774, 860, 698, 748, 596, 527, 734, 38, 237, 107, 699, 363, 221, 462, 920, 797, 554, 95, 650, 549, 823, 466, 19, 249, 115, 508, 16, 490, 305, 576, 761, 719, 75, 399, 710, 314, 781, 690, 232, 201, 933, 634, 535, 657, 54, 12, 178, 838, 253, 891, 353, 825, 376, 866, 598, 0, 144, 81, 859, 498, 828, 854, 613, 10, 766, 908, 840, 574, 35, 448, 852, 672, 39, 847, 587, 485, 30, 914, 909, 83, 189, 524, 316, 101, 70, 836, 428, 191, 890, 551, 518, 892, 844, 162, 215, 548, 611, 79, 181, 646, 796, 486, 639, 640, 425, 285, 559, 745, 17, 102, 361, 210, 912, 356, 355, 385, 89, 222, 345, 923, 378, 507, 240, 152, 636, 394, 849, 56, 726, 662, 349, 164, 901, 630, 208, 176, 266, 28, 863, 49, 129, 348, 131, 934, 635, 656, 775, 126, 799, 392, 155, 52, 418, 206, 51, 684, 384, 372, 87, 919, 575, 370, 88, 556, 331, 340, 510, 667, 550, 711, 172, 500, 590, 757, 404, 332, 821, 263, 149, 786, 663, 783, 881, 442, 4, 565, 558, 642, 715, 342, 853, 756, 409, 77, 480, 32, 236, 306, 855, 758, 280, 205, 235, 936, 547, 297, 588, 532, 125, 118, 251, 37, 276, 247, 604, 885, 910, 388]

        # male_iids = [232, 762,  57, 935, 926,  69, 361, 379, 733, 372,  52, 503, 888,
        #     13, 891, 599,  59, 411, 734, 429, 208, 104, 736, 273, 343, 534,
        # 140, 511, 557, 885, 679, 180, 352,  99, 828,  16, 623, 431, 189,
        # 408, 605, 769, 750, 932, 474, 210, 157, 548, 116, 409, 633, 108,
        # 746, 766, 865, 783, 218, 558, 729, 798, 100, 274, 873, 203, 392,
        # 931, 252, 295, 469, 542, 713, 604, 466, 251, 198, 288, 659, 666,
        # 298, 504,  63, 233,  71, 636, 625, 175, 712, 662, 744, 237,   7,
        # 418, 370, 938, 193, 721, 738, 263, 214, 173, 786, 841, 615, 606,
        # 455, 371,  41, 172,  37, 414, 652, 192, 675, 840, 797, 635, 863,
        # 855,  11, 689, 206, 907, 614, 667, 159, 247, 637, 754, 764, 578,
        # 425, 590, 309, 337,  48, 131, 921, 240, 524,  84, 423, 224, 477,
        #     95, 363, 826, 776, 130, 215, 260, 266, 160, 450, 283,  28, 294,
        # 187, 366, 805, 629, 373, 333, 184, 551,  43, 711, 868, 200,  49,
        # 608, 327, 795, 497,  79,  40,  39, 350, 790, 820, 155,  18,  51,
        # 356, 216, 221, 642, 864, 634, 119, 340, 404, 940, 630, 748, 153,
        # 763, 358, 859, 424, 276, 872,  21, 654, 382,  36,   0,  92, 681,
        # 647, 355, 395, 297, 428, 225, 624, 541, 244, 368, 574,  94, 576,
        # 549,  30, 519, 422, 256, 268,  81, 445, 821, 912, 364, 866,  67,
        # 811, 908, 862, 436, 451, 319, 663, 446, 386, 472, 933, 854,  32,
        # 813, 313, 190, 710, 550, 407, 735, 137, 833, 149, 305, 796, 705,
        #     58, 320, 226, 261, 201, 299, 565, 730, 792, 824, 749, 785, 556]

        # these are 273 most active users
        # male_iids= [452, 7, 641, 296, 797, 690, 46, 474, 548, 638,
        #     859, 719, 688, 75, 741, 167, 706, 554, 214, 559,
        #     370, 384, 678, 633, 414, 484, 22, 275, 934, 898,
        #     269, 52, 697, 385, 535, 418, 472, 48, 864, 894,
        #     203, 567, 300, 659, 866, 469, 11, 154, 575, 732,
        #     283, 87, 689, 204, 603, 200, 226, 636, 168, 463,
        #     261, 490, 394, 9, 25, 32, 909, 29, 97, 785,
        #     225, 122, 792, 766, 422, 380, 19, 352, 101, 833,
        #     114, 412, 171, 460, 408, 578, 517, 69, 662, 30,
        #     483, 545, 822, 723, 910, 647, 51, 102, 89, 448,
        #     729, 504, 147, 349, 881, 1, 702, 505, 274, 47,
        #     197, 291, 240, 456, 434, 299, 18, 406, 823, 596,
        #     20, 53, 815, 193, 348, 931, 215, 192, 368, 940,
        #     17, 347, 194, 435, 698, 180, 284, 305, 652, 3,
        #     855, 21, 820, 579, 379, 24, 79, 112, 63, 93,
        #     119, 551, 714, 233, 395, 471, 617, 353, 721, 317,
        #     407, 658, 748, 760, 83, 107, 813, 388, 478, 849,
        #     591, 189, 570, 399, 581, 176, 455, 673, 208, 458,
        #     59, 423, 44, 202, 415, 511, 49, 569, 892, 337,
        #     116, 475, 173, 205, 184, 667, 343, 345, 288, 623,
        #     790, 558, 84, 555, 661, 672, 784, 687, 8, 386,
        #     552, 174, 210, 209, 557, 206, 933, 853, 922, 62,
        #     185, 100, 747, 268, 149, 162, 223, 477, 130, 315,
        #     289, 876, 625, 438, 378, 108, 754, 77, 705, 35,
        #     713, 811, 70, 4, 285, 857, 593, 585, 889, 834,
        #     485, 532, 862, 904, 786, 493, 278, 920, 498, 34,
        #     392, 751, 366, 818, 798, 271, 773, 355, 878, 140,
        #     38, 216, 938]

        #  for key, val in metric_user_results.items():
        all_mid = []
        male_iids = [62, 371, 114, 579, 368, 143, 47, 538, 769, 807, 460, 473, 942, 207, 180, 469, 806, 461, 339, 212, 1, 616, 940, 861, 534, 90, 8, 439, 638, 134, 53, 120, 187, 170, 25, 408, 609, 824, 681, 367, 669, 44, 426, 477, 749, 689, 275, 755, 813, 438, 14, 566, 737, 614, 774, 860, 698, 748, 596, 527, 734, 38, 237, 107, 699, 363, 221, 462, 920, 797, 554, 95, 650, 549, 823, 466, 19, 249, 115, 508, 16, 490, 305, 576, 761, 719, 75, 399, 710, 314, 781, 690, 232, 201, 933, 634, 535, 657, 54, 12, 178, 838, 253, 891, 353, 825, 376, 866, 598, 0, 144, 81, 859, 498, 828, 854, 613, 10, 766, 908, 840, 574, 35, 448, 852, 672, 39, 847, 587, 485, 30, 914, 909, 83, 189, 524, 316, 101, 70, 836, 428, 191, 890, 551, 518, 892, 844, 162, 215, 548, 611, 79, 181, 646, 796, 486, 639, 640, 425, 285, 559, 745, 17, 102, 361, 210, 912, 356, 355, 385, 89, 222, 345, 923, 378, 507, 240, 152, 636, 394, 849, 56, 726, 662, 349, 164, 901, 630, 208, 176, 266, 28, 863, 49, 129, 348, 131, 934, 635, 656, 775, 126, 799, 392, 155, 52, 418, 206, 51, 684, 384, 372, 87, 919, 575, 370, 88, 556, 331, 340, 510, 667, 550, 711, 172, 500, 590, 757, 404, 332, 821, 263, 149, 786, 663, 783, 881, 442, 4, 565, 558, 642, 715, 342, 853, 756, 409, 77, 480, 32, 236, 306, 855, 758, 280, 205, 235, 936, 547, 297, 588, 532, 125, 118, 251, 37, 276, 247, 604, 885, 910, 388]
        

        # print("*" * 10)
        for k, v in metric_user_results.items():
            m_ids = []
            female_res = []
            male_res = []
            female_iids = []
            print("_" * 10)
            # print(v.items().__len__())
            print(k)
            print("_" * 10)
            for uid, m in v.items():
                # print(f"uid: {uid}")
                # print(f"g: {genders[uid]}")
                g = genders[uid]
                if g == 1 :  # female
                    female_res.append(m)
                    female_iids.append(uid)
                elif g == 0 and uid in male_iids:
                    male_res.append(m)
                    m_ids.append(uid)
            print("#" * 10)
            print("male" * 10)
            print(m_ids)
            print(m_ids.__len__())
            print("female" * 10)
            print(female_iids)
            print(female_iids.__len__())
            print("#" * 10)
            print(f"{sum(female_res) / len(female_res)} female")
            print(f"{sum(male_res) / len(male_res)} male")

        # for k, v in metric_user_results.items():

        #     for uid, m in v.items():
        #         # print(f"uid: {uid}")
        #         # print(f"g: {genders[uid]}")
        #         g = genders[uid]
        #         if g == 1 and uid in female_active:  # female
        #             female_res.append(m)
        #             female_iids.append(uid)
        #         elif g == 0 and uid in male_active:
        #             male_res.append(m)
        #             m_ids.append(uid)

        # print("#" * 10)
        # print("male" * 10)
        # print(m_ids)
        # print(m_ids.__len__())
        # print("female" * 10)
        # print(female_iids)
        # print(female_iids.__len__())
        # print("#" * 10)
        # print(male_iids.__len__())
        # print(female_iids.__len__())
        # print("-"*10)
        # print(male_res.__len__())
        # print(female_res.__len__())

        # print("-"*10)

        # print()
        # print(f"{sum(female_res) / len(female_res)} female")
        # print(f"{sum(male_res) / len(male_res)} male")
        # print(",,,,,,,,")

        return Result(model.name, metric_avg_results, metric_user_results)

    def evaluate(self, model, metrics, user_based, show_validation=True):
        """Evaluate given models according to given metrics. Supposed to be called by Experiment.

        Parameters
        ----------
        model: :obj:`cornac.models.Recommender`
            Recommender model to be evaluated.

        metrics: :obj:`iterable`
            List of metrics.

        user_based: bool, required
            Evaluation strategy for the rating metrics. Whether results
            are averaging based on number of users or number of ratings.

        show_validation: bool, optional, default: True
            Whether to show the results on validation set (if exists).

        Returns
        -------
        res: :obj:`cornac.experiment.Result`
        """
        if self.train_set is None:
            raise ValueError("train_set is required but None!")
        if self.test_set is None:
            raise ValueError("test_set is required but None!")

        self._reset()

        ###########
        # FITTING #
        ###########
        if self.verbose:
            print("\n[{}] Training started!".format(model.name))

        start = time.time()
        model.fit(self.train_set, self.val_set)
        train_time = time.time() - start

        ##############
        # EVALUATION #
        ##############
        if self.verbose:
            print("\n[{}] Evaluation started!".format(model.name))

        rating_metrics, ranking_metrics = self.organize_metrics(metrics)

        start = time.time()
        model.transform(self.test_set)
        test_result = self.eval(
            model=model,
            train_set=self.train_set,
            test_set=self.test_set,
            val_set=self.val_set,
            rating_threshold=self.rating_threshold,
            exclude_unknowns=self.exclude_unknowns,
            rating_metrics=rating_metrics,
            ranking_metrics=ranking_metrics,
            user_based=user_based,
            verbose=self.verbose,
        )
        test_time = time.time() - start
        test_result.metric_avg_results["Train (s)"] = train_time
        test_result.metric_avg_results["Test (s)"] = test_time

        val_result = None
        if show_validation and self.val_set is not None:
            start = time.time()
            model.transform(self.val_set)
            val_result = self.eval(
                model=model,
                train_set=self.train_set,
                test_set=self.val_set,
                val_set=None,
                rating_threshold=self.rating_threshold,
                exclude_unknowns=self.exclude_unknowns,
                rating_metrics=rating_metrics,
                ranking_metrics=ranking_metrics,
                user_based=user_based,
                verbose=self.verbose,
            )
            val_time = time.time() - start
            val_result.metric_avg_results["Time (s)"] = val_time

        return test_result, val_result

    @classmethod
    def from_splits(
        cls,
        train_data,
        test_data,
        val_data=None,
        fmt="UIR",
        rating_threshold=1.0,
        exclude_unknowns=False,
        seed=None,
        verbose=False,
        **kwargs,
    ):
        """Constructing evaluation method given data.

        Parameters
        ----------
        train_data: array-like
            Training data

        test_data: array-like
            Test data

        val_data: array-like, optional, default: None
            Validation data

        fmt: str, default: 'UIR'
            Format of the input data. Currently, we are supporting:

            'UIR': User, Item, Rating
            'UIRT': User, Item, Rating, Timestamp

        rating_threshold: float, default: 1.0
            Threshold to decide positive or negative preferences.

        exclude_unknowns: bool, default: False
            Whether to exclude unknown users/items in evaluation.

        seed: int, optional, default: None
            Random seed for reproduce the splitting.

        verbose: bool, default: False
            The verbosity flag.

        Returns
        -------
        method: :obj:`<cornac.eval_methods.BaseMethod>`
            Evaluation method object.

        """
        method = cls(
            fmt=fmt,
            rating_threshold=rating_threshold,
            exclude_unknowns=exclude_unknowns,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )

        return method.build(
            train_data=train_data, test_data=test_data, val_data=val_data
        )
