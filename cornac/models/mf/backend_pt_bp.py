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

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
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

        self.u_factors = nn.Embedding(*u_factors.shape)
        self.i_factors = nn.Embedding(*i_factors.shape)
        self.u_factors.weight.data = torch.from_numpy(u_factors)
        self.i_factors.weight.data = torch.from_numpy(i_factors)
        if use_bias:
            self.u_biases = nn.Embedding(*u_biases.shape)
            self.i_biases = nn.Embedding(*i_biases.shape)
            self.u_biases.weight.data = torch.from_numpy(u_biases)
            self.i_biases.weight.data = torch.from_numpy(i_biases)

    def forward(self, uids, iids):
        ues = self.u_factors(uids)
        ies = self.i_factors(iids)

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
    criteria = nn.MSELoss(reduction="mean")
    optimizer = OPTIMIZER_DICT[optimizer](
        params=model.parameters(), lr=learning_rate, weight_decay=reg
    )
    new_loss = GenderMseLossBeyonParity(a=alpha, reduction="sum")
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
            g_batch = torch.tensor(model.user_gender[u_batch]).to(device)
            cat_batch = torch.tensor(model.item_cat[i_batch]).to(device)

            preds = model(u_batch, i_batch)
            # loss = criteria(preds, r_batch)
            # print(r_batch.shape)
            if _ == n_epochs:
                # print the max gloss and mseloss for normalization later
                printLoss = True
            loss = new_loss(
                preds,
                r_batch,
                g_batch,
                u_batch,
                i_batch,
                model.item_cat,
                # cat_batch,
                recommender,
                top_k,
                batch_size,
                train_set.max_rating,
                train_set.min_rating,
                printLoss,
            )
            # print(loss.requires_grad)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            all_loss.append(loss.data.item())
            sum_loss += loss.data.item()
            count += len(u_batch)

            if batch_id % 10 == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))
        if printLoss:
            print(all_loss)


class GenderMseLossBeyonParity(nn.MSELoss):
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
        min_rating,
        printLoss=False,
    ):
        mse_loss = super().forward(preds, r_batch)
        f = g_batch == 1  # bool array is t when female
        m = g_batch == 0

        unique_items = torch.unique(i_batch)

        U_val = 0
        for i in unique_items:
            current_item_ind = i_batch == i
            female = f & current_item_ind
            male = m & current_item_ind

            E_g_yj = (
                preds[female].mean() if female.any() else 0
            )  # female avg pred score
            E_mg_yj = preds[male].mean() if male.any() else 0  # male avg pred score
            E_g_rj = (
                r_batch[female].mean() if female.any() else 0
            )  # female avg rating (actual)
            E_mg_rj = (
                r_batch[male].mean() if male.any() else 0
            )  # male avg rating (actual)
            U_val = U_val + abs((E_g_yj - E_mg_yj) - (E_g_rj - E_mg_rj))

        if self.a != 0:
            gender_loss = U_val / len(u_batch)

            loss = (
                self.a  # a is 1
                * (gender_loss / (2 * (max_rating - min_rating)))  # normalize
                * (
                    batch_size * (max_rating - min_rating) ** 2
                )  # scale up to mseloss's scale
                + mse_loss
            )
            # print(
            #     f"loss {loss} mseloss{mse_loss} gend {  (gender_loss / (2 * max_rating)) * (batch_size * (max_rating - min_rating) ** 2) }"
            # )

        else:
            loss = mse_loss

        return loss
