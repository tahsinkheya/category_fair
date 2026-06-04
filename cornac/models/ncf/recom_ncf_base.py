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


import numpy as np
from tqdm.auto import trange
import torch
from glob import glob
from ..recommender import Recommender
from ...utils import get_rng
from ...exception import ScoreException
import os
import pickle

import json
import warnings
from datetime import datetime
from cornac.gender_regularization.GenderLoss import GenderLoss
# from cornac.gender_regularization.GenderLossRCR import GenderLossRCR



class NCFBase(Recommender):
    """Base class of NCF family.

    Parameters
    ----------
    num_epochs: int, optional, default: 20
        Number of epochs.

    batch_size: int, optional, default: 256
        Batch size.

    num_neg: int, optional, default: 4
        Number of negative instances to pair with a positive instance.

    lr: float, optional, default: 0.001
        Learning rate.

    learner: str, optional, default: 'adam'
        Specify an optimizer: adagrad, adam, rmsprop, sgd
    
    backend: str, optional, default: 'tensorflow'
        Backend used for model training: tensorflow, pytorch

    early_stopping: {min_delta: float, patience: int}, optional, default: None
        If `None`, no early stopping. Meaning of the arguments: 
        
         - `min_delta`: the minimum increase in monitored value on validation set to be considered as improvement, \
           i.e. an increment of less than min_delta will count as no improvement.
         - `patience`: number of epochs with no improvement after which training should be stopped.

    name: string, optional, default: 'NCF'
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    References
    ----------
    * He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. \
    In Proceedings of the 26th international conference on world wide web (pp. 173-182).
    """

    def __init__(
        self,
        name="NCF",
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        lr=0.001,
        learner="adam",
        backend="tensorflow",
        early_stopping=None,
        trainable=True,
        verbose=True,
        user_features=None,
        item_features=None,
        top_k=0,
        alp=0,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.lr = lr
        self.learner = learner
        self.backend = backend
        self.early_stopping = early_stopping
        self.user_features = user_features
        self.item_features = item_features
        self.alp = alp
        self.top_k = top_k
        self.seed = seed
        self.rng = get_rng(seed)
        self.ignored_attrs.extend(
            [
                "graph",
                "user_id",
                "item_id",
                "labels",
                "interaction",
                "prediction",
                "loss",
                "train_op",
                "initializer",
                "saver",
                "sess",
            ]
        )

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

        if self.trainable:
            self.num_users = self.num_users
            self.num_items = self.num_items

            if self.backend == "tensorflow":
                self._fit_tf(train_set, val_set)
            elif self.backend == "pytorch":
                self._fit_pt(train_set, val_set)
            else:
                raise ValueError(f"{self.backend} is not supported")

        return self

    ########################
    ## TensorFlow backend ##
    ########################
    def _build_graph_tf(self):
        raise NotImplementedError()

    def _sess_init_tf(self):
        import tensorflow.compat.v1 as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.initializer)

    def _get_feed_dict(self, batch_users, batch_items, batch_ratings):
        return {
            self.user_id: batch_users,
            self.item_id: batch_items,
            self.labels: batch_ratings.reshape(-1, 1),
        }

    def _fit_tf(self, train_set, val_set):
        if not hasattr(self, "graph"):
            self._build_graph_tf()

        loop = trange(self.num_epochs, disable=not self.verbose)
        for _ in loop:
            count = 0
            sum_loss = 0
            for i, (batch_users, batch_items, batch_ratings) in enumerate(
                train_set.uir_iter(
                    self.batch_size, shuffle=True, binary=True, num_zeros=self.num_neg
                )
            ):
                _, _loss = self.sess.run(
                    [self.train_op, self.loss],
                    feed_dict=self._get_feed_dict(
                        batch_users, batch_items, batch_ratings
                    ),
                )
                count += len(batch_users)
                sum_loss += len(batch_users) * _loss
                if i % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))

            if self.early_stopping is not None and self.early_stop(
                train_set, val_set, **self.early_stopping
            ):
                break
        loop.close()

    def _score_tf(self, user_idx, item_idx):
        raise NotImplementedError()

    #####################
    ## PyTorch backend ##
    #####################
    def _build_model_pt(self):
        raise NotImplementedError()

    def _fit_pt(self, train_set, val_set):
        import torch
        import torch.nn as nn
        from .backend_pt import optimizer_dict

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        self.model = self._build_model_pt().to(self.device)

        optimizer = optimizer_dict[self.learner](
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.reg,
        )
        criteria = nn.BCELoss()
        # criteria1 = nn.BCELoss(reduction="sum")
        criteria1 = nn.BCELoss(reduction="none")
        all_loss = []

        loop = trange(self.num_epochs, disable=not self.verbose)
        for _ in loop:
            count = 0
            sum_loss = 0
            gender_values = torch.tensor(
                self.user_features, dtype=torch.float32, requires_grad=True
            ).to(device)
            genre_values = torch.tensor(
                self.item_features, dtype=torch.float32, requires_grad=True
            ).to(device)

            for batch_id, (batch_users, batch_items, batch_ratings) in enumerate(
                train_set.uir_iter(
                    self.batch_size, shuffle=True, binary=True, num_zeros=self.num_neg
                )
            ):
                batch_users = torch.from_numpy(batch_users).to(self.device)
                batch_items = torch.from_numpy(batch_items).to(self.device)
                batch_ratings = torch.tensor(batch_ratings, dtype=torch.float).to(
                    self.device
                )

                optimizer.zero_grad()
                outputs = self.model(batch_users, batch_items)

                if self.alp != 0:
                    # calculate adn add gender loss
                    bce_loss_none = criteria1(outputs, batch_ratings)
                    g_loss = GenderLossRCR(
                        gender=gender_values,
                        users=batch_users,
                        genres=genre_values,
                        recommender=self,
                        top_k=self.top_k,
                    )
                    g_loss = g_loss.compute()
                    g_loss = torch.sigmoid(0.1 * (g_loss - 0.5))

                    bce_loss = criteria(outputs, batch_ratings)
                    # print(self.alp)
                    loss = (
                        self.alp * g_loss * max(bce_loss_none)
                        + (1 - self.alp) * bce_loss
                    )
                    # print(
                    #     f"gloss {g_loss} Bce {bce_loss} loss {loss}"
                    # )

                    # print(
                    #     f"loss{loss} gloss{g_loss} n_gloss {g_loss * max(bce_loss_none)} bceloss {bce_loss} maxbc {max(bce_loss_none)}"
                    # )

                else:
                    loss = criteria(outputs, batch_ratings)
                    # print(loss)

                # print(f"loss {loss}")
                # print(f"loss1 {loss1}")
                # print(f"mean loss {loss1/len(batch_ratings)}")
                # print(f"loss2 {sum(loss)/len(batch_ratings)}")
                loss.backward()
                optimizer.step()
                all_loss.append(loss.data.item())

                count += len(batch_users)
                sum_loss += len(batch_users) * loss.data.item()

                if batch_id % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))

            if self.early_stopping is not None and self.early_stop(
                train_set, val_set, min_delta=0.0005, patience=20
            ):
                break

            if _ == self.num_epochs - 1:
                print(all_loss)
        loop.close()

    def _score_pt(self, user_idx, item_idx):
        raise NotImplementedError()

    def save(self, save_dir=None, metadata=None, save_trainset=True):
        """Save a recommender model to the filesystem.

        Parameters
        ----------
        save_dir: str, default: None
            Path to a directory for the model to be stored.

        """
       
        model_dir = os.path.join(save_dir, self.name)
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        model_file = os.path.join(model_dir, "{}.pkl".format(timestamp))
        if self.backend == "pytorch":
            # Save PyTorch model (state_dict) and optionally other info
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),  # Model parameters
                    # "optimizer_state_dict": self.optimizer.state_dict(),  # Optimizer state
                    # "epoch": self.epoch,  # Epoch (optional)
                    "metadata": metadata,  # Additional metadata
                },
                model_file.replace(".pkl", ".pt"),
            )  # Save as .pt for PyTorch
            if self.verbose:
                print(
                    f"{self.name} PyTorch model is saved to {model_file.replace('.pkl', '.pt')}"
                )
        metadata = {} if metadata is None else metadata
        metadata["model_classname"] = type(self).__name__
        metadata["model_file"] = os.path.basename(model_file)

        if save_trainset:
            trainset_file = model_file + ".trainset"
            pickle.dump(
                self.train_set,
                open(trainset_file, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            metadata["trainset_file"] = os.path.basename(trainset_file)

        # Save metadata to a .meta file
        with open(model_file + ".meta", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        # model_file = Recommender.save(self, save_dir)

        # if self.backend == "tensorflow":
        #     self.saver.save(self.sess, model_file.replace(".pkl", ".cpt"))
        # elif self.backend == "pytorch":
        #     # TODO: implement model saving for PyTorch
        #     # raise NotImplementedError()
        #     torch.save({
        #     'model_state_dict': self.model.state_dict(),  # Save model parameters
        #     'optimizer_state_dict': self.optimizer.state_dict(),  # Save optimizer state (optional)
        #     'epoch': self.epoch,  # Optionally save current epoch
        #     'metadata': metadata  # Optionally save metadata passed from save function
        #     }, model_file.replace(".pkl", ".pt"))

        return model_file

    # @staticmethod
    def load(self, model_path, trainable=False, name="name"):
        """Load a recommender model from the filesystem.

        Parameters
        ----------
        model_path: str, required
            Path to a file or directory where the model is stored. If a directory is
            provided, the latest model will be loaded.

        trainable: boolean, optional, default: False
            Set it to True if you would like to finetune the model. By default,
            the model parameters are assumed to be fixed after being loaded.

        Returns
        -------
        self : object
        """
        if os.path.isdir(model_path):
            # Pick the latest saved model file in the directory
            model_file = sorted(glob(os.path.join(model_path, "*.[pkl|pt]")))[-1]
        else:
            model_file = model_path
        if model_file.endswith(".pt"):
            checkpoint = torch.load(model_file)

            model = self.model  # Instantiate the class (or use an existing model class)
            model.load_state_dict(checkpoint["model_state_dict"])

            if trainable:
                # If training is desired, the optimizer must also be restored
                model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Optionally load additional information like epoch or metadata
            model.epoch = checkpoint.get("epoch", None)
            model.metadata = checkpoint.get("metadata", {})
            model.trainable = trainable
            model.load_from = model_file

        else:
            # Handle non-PyTorch models (default to pickle loading)
            model = pickle.load(open(model_file, "rb"))
            model.trainable = trainable
            model.load_from = model_file
        # model = Recommender.load(model_path, trainable)
        # if hasattr(model, "pretrained"):  # NeuMF
        #     model.pretrained = False

        # if model.backend == "tensorflow":
        #     model._build_graph()
        #     model.saver.restore(model.sess, model.load_from.replace(".pkl", ".cpt"))
        # elif model.backend == "pytorch":
        #     # TODO: implement model loading for PyTorch

        return model

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

        n = ranking_eval(
            model=self,
            metrics=[NDCG(k=20)],
            train_set=train_set,
            test_set=val_set,
        )[0][0]
        # print(f"hit ratio {hr}")

        return n

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
        if self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)

        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        if self.backend == "tensorflow":
            pred_scores = self._score_tf(user_idx, item_idx)
        elif self.backend == "pytorch":
            pred_scores = self._score_pt(user_idx, item_idx)

        return pred_scores.ravel()
