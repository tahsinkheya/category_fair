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
## part of this code is follwing the implementation of # 
# @Author : Jiakai Tang, Zihan Lin, Yupeng Hou, Yushuo Chen, Shanlei Mu, Xingyu Pan
# Shanlei Mu ,Hui Wang, Xinyan Fan, Chen Yang, Yibo Li, Lanling Xu, Haoran Cheng, Zhichao Feng
#from recbole https://recbole.io/quickstart.html

import torch
import torch.nn as nn
from tqdm.auto import trange
from cornac.gender_regularization.GenderLoss import GenderLossMF
import numpy as np

OPTIMIZER_DICT = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}


class MF(nn.Module):
    def __init__(
        self,
        u_factors,
        i_factors,
        filter_layer,
        dis_layer_dict,
        u_biases,
        i_biases,
        use_bias,
        global_mean,
        global_mean_implicit,
        dropout,
        user_gender=None,
        item_cat=None,
    ):
        super(MF, self).__init__()

        self.use_bias = use_bias
        self.global_mean = global_mean
        self.global_mean_implicit = global_mean_implicit
        self.dropout = nn.Dropout(p=dropout)
        self.user_gender = user_gender
        self.item_cat = item_cat
        self.filter_layer = filter_layer
        self.dis_layer_dict = dis_layer_dict

        self.u_factors = nn.Embedding(*u_factors.shape)
        self.i_factors = nn.Embedding(*i_factors.shape)
        self.u_factors.weight.data = torch.from_numpy(u_factors)
        self.i_factors.weight.data = torch.from_numpy(i_factors)
        if use_bias:
            self.u_biases = nn.Embedding(*u_biases.shape)
            self.i_biases = nn.Embedding(*i_biases.shape)
            self.u_biases.weight.data = torch.from_numpy(u_biases)
            self.i_biases.weight.data = torch.from_numpy(i_biases)
            
        self.optimizer_filter = self._build_optimizer(
                params=[{"params": self.u_factors.weight}]
                + [{"params": self.i_factors.weight}]
                + [{"params": _.parameters()} for _ in self.filter_layer.values()]
                + [{"params": self.u_biases.weight}]
                + [{"params": self.i_biases.weight}]
                + [{"params": self.model.global_bias}]
            )
        self.optimizer_dis = self._build_optimizer(
            params=[
                {"params": _.parameters()} for _ in self.dis_layer_dict.values()
            ]
        )
    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop("params", self.model.parameters())
        # learner = kwargs.pop("learner", self.learner)
        learning_rate = kwargs.pop("learning_rate", self.learning_rate)
        weight_decay = kwargs.pop("weight_decay", self.weight_decay)

        if (
            self.config["reg_weight"]
            and weight_decay
            and weight_decay * self.config["reg_weight"] > 0
        ):
            self.logger.warning(
                "The parameters [weight_decay] and [reg_weight] are specified simultaneously, "
                "which may lead to double regularization."
            )

        optimizer = OPTIMIZER_DICT[optimizer](params, lr=learning_rate, weight_decay=weight_decay)
       
        return optimizer

    def forward(self, uids, iids):
        ues = self.u_factors(uids)
        ies = self.i_factors(iids)
        user_embed = self.filter_layer[0](ues)
        

        preds = (self.dropout(ues) * self.dropout(ies)).sum(dim=1, keepdim=True)
        if self.use_bias:
            preds += self.u_biases(uids) + self.i_biases(iids) + self.global_mean

        return preds.squeeze()


def learn(
    model,
    train_set,
    recommender,
    top_k,
    n_epochs,
    batch_size=256,
    learning_rate=0.01,
    reg=1e-5,
    verbose=True,
    optimizer="sgd",
    device=torch.device("cpu"),
    alpha=0,
):
    model = model.to(device)
    optimizer = OPTIMIZER_DICT[optimizer](
        params=model.parameters(), lr=learning_rate, weight_decay=reg
    )
    new_loss = GenderMseLoss(a=alpha, reduction="none")
    printLoss = False
    all_loss = []
    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    for _ in progress_bar:
        sum_loss = 0.0
        count = 0
        for batch_id, (u_batch, i_batch, r_batch) in enumerate(
            train_set.uir_iter(batch_size, shuffle=True)
        ):
            u_batch = torch.from_numpy(u_batch).to(device)
            i_batch = torch.from_numpy(i_batch).to(device)
            r_batch = torch.tensor(r_batch, dtype=torch.float).to(device)

            preds = model(u_batch, i_batch)

            # loss = criteria(preds, r_batch)
            # print(r_batch.shape)
            if _ == n_epochs:
                printLoss = True
            loss = new_loss(
                preds,
                r_batch,
                torch.tensor(model.user_gender).to(device),
                u_batch,
                i_batch,
                torch.tensor(model.item_cat).to(device),
                # cat_batch,
                recommender,
                top_k,
                batch_size,
                train_set.max_rating,
                printLoss,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            recommender.u_factors_batch = model.u_factors.weight.squeeze()

            recommender.i_factors_batch = model.i_factors.weight.squeeze()
            recommender.u_biases_batch = model.u_biases.weight.squeeze()

            recommender.i_biases_batch = model.i_biases.weight.squeeze()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            all_loss.append(loss.data.item())
            sum_loss += loss.data.item()
            count += len(u_batch)

            if batch_id % 10 == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))
        if printLoss:
            print(all_loss)


class GenderMseLoss(nn.MSELoss):
    def __init__(self, a, reduction):
        super().__init__(reduction=reduction)
        self.a = a

    def forward(
        self,
        preds,
        r_batch,
        g_batch,
        u_batch,
        i_batch,
        genres,
        recommender,
        top_k,
        batch_size,
        max_rating,
        printLoss=False,
    ):
        mse_loss = super().forward(preds, r_batch)
        f = g_batch == 1
        m = g_batch == 0
        diff = torch.abs(r_batch - preds)

        # equation 3 start____________________________
        gender_loss = 0
        if self.a != 0:
            glmf = GenderLossMF(g_batch, u_batch, genres, recommender, top_k)
            gender_loss = glmf.compute()

            loss = (
                self.a
                * gender_loss
                * (
                    batch_size * max(mse_loss)
                )  # scale up gender loss need to multipply with batchsize bcoz we are using reduction sum for the mseloss below
                + (1 - self.a) * mse_loss.sum()  # total batch loss
            )
        else:
            loss = mse_loss.sum()

        # glmf = GenderLossMF(
        #         g_batch, u_batch, i_batch, diff, genres, recommender, top_k
        #     )

        # print(
        #     f"{type(loss)} {type(mse_loss.sum())} { type(gender_loss * (batch_size * max(mse_loss)))}"
        # )
        # print(
        #     f"loss {loss}  mse_loss {mse_loss.sum()} gloss {gender_loss * (batch_size * max(mse_loss))} pure gloss {gender_loss}"
        # )
        # print(mse_loss)

        return loss
