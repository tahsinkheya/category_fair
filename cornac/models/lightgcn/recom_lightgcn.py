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

from ..recommender import Recommender
from ..recommender import ANNMixin, MEASURE_DOT
from ...exception import ScoreException
import numpy as np
from cornac.models.lightgcn.GenderLossLGCN import GenderLossGCN
import torch
from tqdm.auto import tqdm, trange


class LightGCN(Recommender, ANNMixin):
    """
    LightGCN

    Parameters
    ----------
    name: string, default: 'LightGCN'
        The name of the recommender model.

    emb_size: int, default: 64
        Size of the node embeddings.

    num_epochs: int, default: 1000
        Maximum number of iterations or the number of epochs.

    learning_rate: float, default: 0.001
        The learning rate that determines the step size at each iteration

    batch_size: int, default: 1024
        Mini-batch size used for train set

    num_layers: int, default: 3
        Number of LightGCN Layers

    early_stopping: {min_delta: float, patience: int}, optional, default: None
        If `None`, no early stopping. Meaning of the arguments:

        - `min_delta`:  the minimum increase in monitored value on validation
                        set to be considered as improvement,
                        i.e. an increment of less than min_delta will count as
                        no improvement.

        - `patience`:   number of epochs with no improvement after which
                        training should be stopped.

    lambda_reg: float, default: 1e-4
        Weight decay for the L2 normalization

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model
        is already pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: 2020
        Random seed for parameters initialization.

    References
    ----------
    *   He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020).
        LightGCN: Simplifying and Powering Graph Convolution Network for
        Recommendation.
    """

    def __init__(
        self,
        name="LightGCN",
        emb_size=64,
        num_epochs=1000,
        learning_rate=0.001,
        batch_size=256,
        num_layers=3,
        early_stopping=None,
        lambda_reg=1e-4,
        trainable=True,
        user_features=None,
        item_features=None,
        verbose=False,
        seed=2020,
        alpha=0,
        top_k=0,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.emb_size = emb_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.early_stopping = early_stopping
        self.lambda_reg = lambda_reg
        self.seed = seed
        self.user_features = user_features
        self.item_features = item_features
        self.alpha = alpha
        self.top_k = top_k

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        gender_values = np.array(list(train_set.uid_gender_map.values()))
        item_cats = np.array(list(train_set.iid_cat_map.values()))
        self.user_features = gender_values
        self.item_features = item_cats
        Recommender.fit(self, train_set, val_set)

        if not self.trainable:
            return self

        # model setup
        import torch
        from .lightgcn import Model
        from .lightgcn import construct_graph

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        graph = construct_graph(train_set, self.total_users, self.total_items).to(
            device
        )
        # print(":::::")
        # print(graph)
        # print(":::::")
        model = Model(
            graph,
            self.emb_size,
            self.num_layers,
            self.lambda_reg,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # model training
        pbar = trange(
            self.num_epochs,
            desc="Training",
            unit="iter",
            position=0,
            leave=False,
            disable=not self.verbose,
        )

        for _ in pbar:
            model.train()
            accum_loss = 0.0
            for batch_u, batch_i, batch_j in tqdm(
                train_set.uij_iter(
                    batch_size=self.batch_size,
                    shuffle=True,
                ),
                desc="Epoch",
                total=train_set.num_batches(self.batch_size),
                leave=False,
                position=1,
                disable=not self.verbose,
            ):

                (
                    u_g_embeddings,
                    pos_i_g_embeddings,
                    neg_i_g_embeddings,
                    all_u_emb,
                    all_i_emb,
                ) = model(graph, batch_u, batch_i, batch_j)
                self.U_in_batch = all_u_emb
                self.V_in_batch = all_i_emb

                batch_loss, batch_bpr_loss, batch_reg_loss, all_batch_loss = (
                    model.loss_fn(
                        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
                    )
                )
                # print(f"batch loss {batch_loss}")
                accum_loss += batch_loss.cpu().item() * len(batch_u)
                u_batch = torch.from_numpy(batch_u).to(device)

                g_batch = torch.tensor(self.user_features[u_batch]).to(device)

                if self.alpha != 0:
                    gl = GenderLossGCN(
                        gender=g_batch,
                        users=u_batch,
                        genres=self.item_features,
                        recommender=self,
                        top_k=self.top_k,
                    )
                    gender_loss = gl.compute()
                else:
                    gender_loss = 0

                loss = (
                    self.alpha * gender_loss * max(all_batch_loss)
                    + (1 - self.alpha) * batch_loss
                )

                # print(f"gender loss {gender_loss } batch loss {batch_loss} loss {loss}")
                # print(
                #     f"gender loss norma{gender_loss * max(all_batch_loss)} batch loss {batch_loss} loss {loss}"
                # )
                optimizer.zero_grad()
                loss.backward()

                # for name, param in model.named_parameters():
                #     print(name)
                #     if param.grad is not None:
                #         print(param.grad.norm())
                optimizer.step()

            accum_loss /= len(train_set.uir_tuple[0])  # normalize over all observations
            pbar.set_postfix(loss=accum_loss)

            # store user and item embedding matrices for prediction
            model.eval()
            u_embs, i_embs, _, _, _ = model(graph)

            # we will use numpy for faster prediction in the score function, no need torch
            self.U = u_embs.cpu().detach().numpy()

            self.V = i_embs.cpu().detach().numpy()

            if self.early_stopping is not None and self.early_stop(
                train_set, val_set, **self.early_stopping
            ):
                break

    def monitor_value(self, train_set, val_set):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        res : float
            Monitored value on validation set.
            Return `None` if `val_set` is `None`.
        """
        if val_set is None:
            return None

        from ...metrics import Recall
        from ...eval_methods import ranking_eval

        recall_20 = ranking_eval(
            model=self,
            metrics=[Recall(k=20)],
            train_set=train_set,
            test_set=val_set,
        )[0][0]

        return recall_20  # Section 4.1.2 in the paper, same strategy as NGCF.

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_idx is None:
            if not self.knows_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )
            known_item_scores = self.V.dot(self.U[user_idx, :])
            return known_item_scores
        else:
            if not (self.knows_user(user_idx) and self.knows_item(item_idx)):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            return self.V[item_idx, :].dot(self.U[user_idx, :])

    def get_vector_measure(self):
        """Getting a valid choice of vector measurement in ANNMixin._measures.

        Returns
        -------
        measure: MEASURE_DOT
            Dot product aka. inner product
        """
        return MEASURE_DOT

    def get_user_vectors(self):
        """Getting a matrix of user vectors serving as query for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of user vectors for all users available in the model.
        """
        return self.U

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of item vectors for all items available in the model.
        """
        return self.V

    def score_edited(self, user_idx, item_idx=None):
        """
        ADDED FOR PAPER EQUAL LIGHTS, FAIR CAMERA, DIVERSE ACTIONS!

        "Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.
        self.V_in_batch: represents inbatch embeddings for the current batch, its diff to self.V
        self.U_in_batch: represents inbatch embeddings for the current batch, its diff to self.U

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_idx is None:
            if not self.knows_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )
            known_item_scores = (
                self.V_in_batch.detach()
                .numpy()
                .dot(self.U_in_batch.detach().numpy()[user_idx, :])
            )
            # known_item_scores = torch.matmul(
            #     self.V_in_batch, self.U_in_batch[user_idx, :]
            # )

            return known_item_scores
        else:
            if not (self.knows_user(user_idx) and self.knows_item(item_idx)):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            return self.V_in_batch[item_idx, :].dot(self.U_in_batch[user_idx, :])

    def rank_edited(self, user_idx, item_indices=None, k=-1, **kwargs):
        """
        ADDED FOR PAPER EQUAL LIGHTS, FAIR CAMERA, DIVERSE ACTIONS!
        Rank all test items for a given user using the new score method

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform item raking.

        item_indices: 1d array, optional, default: None
            A list of candidate item indices to be ranked by the user.
            If `None`, list of ranked known item indices and their scores will be returned.

        k: int, required
            Cut-off length for recommendations, k=-1 will return ranked list of all items.
            This is more important for ANN to know the limit to avoid exhaustive ranking.

        Returns
        -------
        (ranked_items, item_scores): tuple
            `ranked_items` contains item indices being ranked by their scores.
            `item_scores` contains scores of items corresponding to index in `item_indices` input.

        """
        # obtain item scores from the model
        try:
            known_item_scores = self.score_edited(user_idx, **kwargs)
        except ScoreException:
            known_item_scores = np.ones(self.total_items) * self.default_score()

        # check if the returned scores also cover unknown items
        # if not, all unknown items will be given the MIN score

        if len(known_item_scores) == self.total_items:
            all_item_scores = known_item_scores
        else:
            all_item_scores = np.ones(self.total_items) * np.min(known_item_scores)
            all_item_scores[: self.num_items] = known_item_scores

        # rank items based on their scores
        item_indices = (
            np.arange(self.num_items)
            if item_indices is None
            else np.asarray(item_indices)
        )
        item_scores = all_item_scores[item_indices]

        if k != -1:  # O(n + k log k), faster for small k which is usually the case
            partitioned_idx = np.argpartition(item_scores, -k)
            top_k_idx = partitioned_idx[-k:]
            sorted_top_k_idx = top_k_idx[np.argsort(item_scores[top_k_idx])]
            partitioned_idx[-k:] = sorted_top_k_idx
            ranked_items = item_indices[partitioned_idx[::-1]]
            # print(ranked_items)
        else:  # O(n log n)
            ranked_items = item_indices[item_scores.argsort()[::-1]]

        return ranked_items, item_scores


#  _, partitioned_idx = torch.topk(item_scores, k, largest=True, sorted=False)
#             sorted_top_k_idx = partitioned_idx[torch.argsort(item_scores[partitioned_idx])]

#             partitioned_idx[-k:] = sorted_top_k_idx
#             ranked_items = item_indices[partitioned_idx.flip(0)]
