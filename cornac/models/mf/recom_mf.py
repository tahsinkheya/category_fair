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


import multiprocessing

import numpy as np
import torch
from ..recommender import Recommender
from ..recommender import ANNMixin, MEASURE_DOT
from ...exception import ScoreException
from ...utils import fast_dot
from ...utils import get_rng
from ...utils.init_utils import normal, zeros


class MF(Recommender, ANNMixin):
    """Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.

    backend: str, optional, default: 'cpu'
        Backend used for model training: cpu, pytorch
    
    optimizer: str, optional, default: 'sgd'
        Specify an optimizer: adagrad, adam, rmsprop, sgd. (ineffective if using CPU backend)
        
    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for training.

    learning_rate: float, optional, default: 0.01
        The learning rate.
        
    batch_size: int, optional, default: 256
        Batch size (ineffective if using CPU backend).

    lambda_reg: float, optional, default: 0.001
        The lambda value used for regularization.
        
    dropout: float, optional, default: 0.0
        The dropout rate of embedding. (ineffective if using CPU backend)

    use_bias: boolean, optional, default: True
        When True, user, item, and global biases are used.

    early_stopping: boolean, optional, default: False
        When True, delta loss will be checked after each iteration to stop learning earlier.

    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization. 
        (Only effective if using CPU backend).

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors,
        'Bu': user_biases, 'Bi': item_biases}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).
        
    run_mode: str, optional, default: genre
        Used to decide which model to run. genre would run the implementation of my paper title "EQUAL LIGHTS, FAIR CAMERA, DIVERSE ACTIONS!"
        bp would run the implementation of paper titled "Beyond Parity: Fairness Objectives for Collaborative Filtering"  url = {https://proceedings.neurips.cc/paper_files/paper/2017/file/e6384711491713d29bc63fc5eeb5ba4f-Paper.pdf},

    References
    ----------
    * Koren, Y., Bell, R., & Volinsky, C. Matrix factorization techniques for recommender systems. \
    In Computer, (8), 30-37. 2009.
    """

    def __init__(
        self,
        name="MF",
        k=10,
        backend="cpu",
        optimizer="sgd",
        max_iter=20,
        learning_rate=0.01,
        batch_size=256,
        lambda_reg=0.02,
        dropout=0.0,
        use_bias=True,
        early_stopping=False,
        num_threads=0,
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
        user_features=None,
        item_features=None,
        alpha=0,
        top_k=0,
        run_mode="genre",
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.backend = backend
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.dropout = dropout
        self.use_bias = use_bias
        self.early_stopping = early_stopping
        self.seed = seed
        self.user_features = user_features
        self.item_features = item_features
        self.alpha = alpha
        self.top_k = top_k
        self.run_mode = run_mode

        if seed is not None:
            self.num_threads = 1
        elif num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.u_factors = self.init_params.get("U", None)
        self.i_factors = self.init_params.get("V", None)
        self.u_biases = self.init_params.get("Bu", None)
        self.i_biases = self.init_params.get("Bi", None)

        # these values are gonna be used to calculate gender loss so we are gonna update these after training every batch
        self.u_factors_batch = self.init_params.get("U", None)
        self.i_factors_batch = self.init_params.get("V", None)
        self.u_biases_batch = self.init_params.get("Bu", None)
        self.i_biases_batch = self.init_params.get("Bi", None)

    def _init(self):
        rng = get_rng(self.seed)

        if self.u_factors is None:
            self.u_factors = normal(
                [self.num_users, self.k], std=0.01, random_state=rng
            )
        if self.i_factors is None:
            self.i_factors = normal(
                [self.num_items, self.k], std=0.01, random_state=rng
            )

        self.u_biases = (
            zeros(self.num_users) if self.u_biases is None else self.u_biases
        )

        self.i_biases = (
            zeros(self.num_items) if self.i_biases is None else self.i_biases
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_mean = self.global_mean if self.use_bias else 0.0
        self.global_mean_implicit = self.global_mean_implicit if self.use_bias else 0.0
        self.u_factors_batch = (
            torch.from_numpy(self.u_factors).to(device).requires_grad_(True)
        )
        self.i_factors_batch = (
            torch.from_numpy(self.i_factors).to(device).requires_grad_(True)
        )
        self.u_biases_batch = (
            torch.from_numpy(self.u_biases).to(device).requires_grad_(True)
        )
        self.i_biases_batch = (
            torch.from_numpy(self.i_biases).to(device).requires_grad_(True)
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

        self._init()

        if self.trainable:
            if self.backend == "cpu":
                self._fit_cpu(train_set, val_set)
            elif self.backend == "pytorch":
                self._fit_pt(train_set, val_set)
            else:
                raise ValueError(f"{self.backend} is not supported")
        return self

    #################
    ## CPU backend ##
    #################
    def _fit_cpu(self, train_set, val_set):
        from cornac.models.mf import backend_cpu

        (rid, cid, val) = train_set.uir_tuple

        backend_cpu.fit_sgd(
            rid,
            cid,
            val.astype(np.float32),
            self.u_factors,
            self.i_factors,
            self.u_biases,
            self.i_biases,
            self.num_users,
            self.num_items,
            self.learning_rate,
            self.lambda_reg,
            self.global_mean,
            self.max_iter,
            self.num_threads,
            self.use_bias,
            self.early_stopping,
            self.verbose,
            self.user_features,
            self.item_features,
        )

    #####################
    ## PyTorch backend ##
    #####################
    def _fit_pt(self, train_set, val_set):
        import torch

        if self.run_mode == "genre":
            from .backend_pt import MF, learn
        else:
            from .backend_pt_bp import MF, learn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        model = MF(
            self.u_factors,
            self.i_factors,
            self.u_biases.reshape(-1, 1),
            self.i_biases.reshape(-1, 1),
            self.use_bias,
            self.global_mean,
            self.global_mean_implicit,
            self.dropout,
            user_gender=self.user_features,
            item_cat=self.item_features,
        )

        learn(
            model=model,
            train_set=train_set,
            val_set=val_set,
            n_epochs=self.max_iter,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            reg=self.lambda_reg,
            optimizer=self.optimizer,
            device=device,
            alpha=self.alpha,
            recommender=self,
            top_k=self.top_k,
            early_stopping=self.early_stopping,
        )
        self.u_factors = model.u_factors.weight.detach().cpu().numpy()
        self.i_factors = model.i_factors.weight.detach().cpu().numpy()
        if self.use_bias:
            self.u_biases = model.u_biases.weight.detach().cpu().squeeze().numpy()
            self.i_biases = model.i_biases.weight.detach().cpu().squeeze().numpy()

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
        # print(">>>>>>")
        # print(self.i_factors[9])
        # print(">>bbbb>>>>")
        # print(self.u_factors[user_idx])
        # print(">>>>>>")
        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        if item_idx is None:
            known_item_scores = self.global_mean + self.i_biases
            if self.knows_user(user_idx):
                known_item_scores += self.u_biases[user_idx]
                # print(self.u_biases[user_idx].shape)
                # print(self.u_factors[user_idx].shape)
                # print(self.i_factors.shape)
                fast_dot(self.u_factors[user_idx], self.i_factors, known_item_scores)
            return known_item_scores
        else:
            item_score = self.global_mean + self.i_biases[item_idx]
            if self.knows_user(user_idx):
                item_score += self.u_biases[user_idx]
                item_score += self.u_factors[user_idx].dot(self.i_factors[item_idx])
            return item_score

    def differentiable_score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.
        this is a special method for paper titled PAPER EQUAL LIGHTS, FAIR CAMERA, DIVERSE ACTIONS!
        to make the funciton differentiable since we are using this for loss optimization


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
        # if item_idx is not None and self.is_unknown_item(item_idx):
        #     raise ScoreException("Can't make score prediction for item %d" % item_idx)
        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        if item_idx is None:
            known_item_scores_2 = self.global_mean + self.i_biases_batch

            if self.knows_user(user_idx):
                known_item_scores_2 += self.u_biases_batch[user_idx]
                known_item_scores_2 += torch.matmul(
                    self.u_factors_batch[user_idx], self.i_factors_batch.T
                )
            return known_item_scores_2
        else:
            item_score_2 = self.global_mean + self.i_biases_batch[item_idx]
            if self.knows_user(user_idx):
                item_score_2 += self.u_biases_batch[user_idx]
                item_score_2 += torch.dot(
                    self.u_factors_batch[user_idx], self.i_factors_batch[item_idx]
                )
            return item_score_2

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
        user_vectors = self.u_factors
        if self.use_bias:
            user_vectors = np.concatenate(
                (user_vectors, np.ones([user_vectors.shape[0], 1])), axis=1
            )
        return user_vectors

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of item vectors for all items available in the model.
        """
        item_vectors = self.i_factors
        if self.use_bias:
            item_vectors = np.concatenate(
                (item_vectors, self.i_biases.reshape((-1, 1))), axis=1
            )
        return item_vectors

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

        from ...metrics import HitRatio
        from ...eval_methods import ranking_eval

        hr = ranking_eval(
            model=self,
            metrics=[HitRatio(k=20)],
            train_set=train_set,
            test_set=val_set,
        )[0][0]

        return hr
