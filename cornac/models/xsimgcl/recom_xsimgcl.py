from ..recommender import Recommender
from ..recommender import ANNMixin, MEASURE_DOT
from ...exception import ScoreException
import numpy as np

# from cornac.models.lightgcn.GenderLossLGCN import GenderLossGCN
from cornac.gender_regularization.GenderLoss import GenderLoss

import torch
from tqdm.auto import tqdm, trange


class XSimGCL(Recommender, ANNMixin):
    def __init__(
        self,
        name="XSimGCL",
        eps=0.2,
        layer_cl=1,
        tau=0.2,  # l* — contrast layer 1 against the final layer
        emb_size=64,
        num_epochs=1000,
        lambda_cl=0.2,  # λ — weight of the InfoNCE loss
        learning_rate=0.001,
        batch_size=256,
        num_layers=3,
        early_stopping=None,
        lambda_reg=1e-4,
        trainable=True,
        user_features=None,
        item_features=None,
        verbose=False,
        seed=42,
        alpha=0,
        top_k=0,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.emb_size = emb_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lambda_reg = lambda_reg
        self.lambda_cl = lambda_cl

        self.eps = eps
        self.tau = tau
        self.layer_cl = layer_cl

        self.early_stopping = early_stopping

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
        from .xsimgcl import Model
        from .xsimgcl import construct_graph

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        graph = construct_graph(train_set, self.total_users, self.total_items).to(
            device
        )
        model = Model(
            graph,
            self.emb_size,
            self.eps,
            self.num_layers,
            self.layer_cl,
            self.lambda_reg,
            self.tau,
        ).to(device)
        #     def __init__(
        #     self,
        #     g,
        #     in_size,
        #     eps,
        #     n_layers,
        #     layer_cl,
        #     lambda_reg,
        #     tau=0.2,
        # ):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
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
                    user_embeds,
                    item_embeds,
                    user_cl_embeddings,
                    item_cl_embeddings,
                ) = model(graph, batch_u, batch_i, batch_j, perturbed=True)
                self.U_in_batch = user_embeds
                self.V_in_batch = item_embeds
                batch_loss, batch_bpr_loss, batch_reg_loss, all_batch_loss = (
                    model.loss_fn(
                        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
                    )
                )

                u_batch = torch.from_numpy(batch_u).to(device)
                if self.alpha != 0:
                    gl = GenderLoss(
                        gender=torch.tensor(self.user_features).to(device),
                        users=u_batch,
                        genres=torch.tensor(self.item_features).to(device),
                        recommender=self,
                        top_k=self.top_k,
                    )
                    gender_loss = gl.compute()
                    gender_loss = torch.sigmoid(0.1 * (gender_loss - 0.5))

                else:
                    gender_loss = 0

                cl_loss = model.cl_loss(
                    batch_u,
                    batch_i,
                    user_embeds,
                    user_cl_embeddings,
                    item_embeds,
                    item_cl_embeddings,
                )
                xsim_loss = batch_loss + self.lambda_cl * cl_loss
                loss = (
                    self.alpha * gender_loss * max(all_batch_loss)
                    + (1 - self.alpha) * xsim_loss
                )
                accum_loss += xsim_loss.cpu().item() * len(batch_u)
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
            accum_loss /= len(train_set.uir_tuple[0])

            pbar.set_postfix(loss=accum_loss)

            model.eval()
            u_embs, i_embs, _, _, _ = model(graph)

            # we will use numpy for faster prediction in the score function, no need torch
            self.U = u_embs.cpu().detach().numpy()

            self.V = i_embs.cpu().detach().numpy()

            if self.early_stopping is not None and self.early_stop(
                train_set, val_set, min_delta=0.0005, patience=20
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

        from ...metrics import NDCG
        from ...eval_methods import ranking_eval

        recall_20 = ranking_eval(
            model=self,
            metrics=[NDCG(k=20)],
            train_set=train_set,
            test_set=val_set,
        )[0][0]

        return recall_20

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

    def differentiable_score(self, user_idx, item_idx=None):
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

            known_item_scores_2 = torch.matmul(
                self.V_in_batch, self.U_in_batch[user_idx, :]
            )

            # known_item_scores_3 = self.V_in_batch.dot(self.U_in_batch[user_idx, :])

            return known_item_scores_2
        else:
            if not (self.knows_user(user_idx) and self.knows_item(item_idx)):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            return torch.dot(
                self.V_in_batch[item_idx, :], (self.U_in_batch[user_idx, :])
            )
