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
import itertools

import torch
from tqdm.auto import tqdm, trange

import torch.nn as nn
from tqdm.auto import trange
from cornac.gender_regularization.GenderLoss import GenderLossMF
import numpy as np
import torch.nn.functional as F
import random

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
        random.seed(123)

        self.use_bias = use_bias
        self.global_mean = global_mean
        self.global_mean_implicit = global_mean_implicit
        self.dropout = nn.Dropout(p=dropout)
        self.user_gender = user_gender
        self.item_cat = item_cat

        self.u_factors = nn.Embedding(*u_factors.shape)
        self.i_factors = nn.Embedding(*i_factors.shape)
        self.u_factors.weight.data = torch.from_numpy(u_factors)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1), requires_grad=True)
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
            preds += self.u_biases(uids) + self.i_biases(iids) + self.global_bias

        # print(self.global_mean_implicit)
        # print("::::")

        return preds.squeeze()


def learn(
    model,
    train_set,
    val_set,
    recommender,
    top_k,
    n_epochs,
    batch_size=256,
    learning_rate=0.01,
    reg=1e-5,
    verbose=True,
    optimizer="sgd",
    device=torch.device("cpu"),
    save_dir=None,
    alpha=0,
    early_stopping=False,
):
    model = model.to(device)
    optimizer = OPTIMIZER_DICT[optimizer](
        params=model.parameters(), lr=learning_rate, weight_decay=reg
    )
    new_loss = TotalLoss(a=alpha)
    # loss_func = BPR_loss_edit(a=alpha)

    printLoss = False
    all_loss = []
    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    genress = torch.tensor(model.item_cat, dtype=torch.float32, requires_grad=True).to(
        device
    )
    genders = torch.tensor(
        model.user_gender, dtype=torch.float32, requires_grad=True
    ).to(device)

    for _ in progress_bar:
        sum_loss = 0.0
        count = 0

        # for batch_id, (u_batch, i_batch, r_batch) in enumerate(
        #     train_set.uir_iter(batch_size, shuffle=True, binary=True, num_zeros=1)
        # ):
        for batch_u, batch_i, batch_j in tqdm(
            train_set.uij_iter(
                batch_size=batch_size,
                shuffle=True,
            ),
            desc="Epoch",
            total=train_set.num_batches(batch_size),
            leave=False,
            position=1,
            disable=not verbose,
        ):
            u_batch = torch.from_numpy(batch_u).to(device)
            i_batch = torch.from_numpy(batch_i).to(device)
            j_batch = torch.from_numpy(batch_j).to(device)
            item_batch = torch.cat((i_batch, j_batch), dim=0)
            g_batch = torch.tensor(model.user_gender[u_batch]).to(device)
            user_batch = torch.cat((u_batch, u_batch), dim=0)

            preds = model(user_batch, item_batch)
            if _ == n_epochs:
                printLoss = True
            # loss = loss_func.compute(preds, batch_size, u_batch)
            loss = new_loss.forward(
                preds,
                g_batch,
                u_batch,
                i_batch,
                genress,
                recommender,
                top_k,
                batch_size,
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

            # if batch_id % 10 == 0:
            progress_bar.set_postfix(loss=(sum_loss / count))

        if early_stopping and recommender.early_stop(
            train_set, val_set, min_delta=0.0005, patience=10
        ):
            print(all_loss)
            break

        if printLoss:
            print(all_loss)


class TotalLoss:
    def __init__(self, a):
        # super().__init__(reduction=reduction)
        self.bpr_loss = BPR_loss_edit()
        # self.xcriteria = nn.BCELoss()

        self.a = a

    def forward(
        self,
        preds,
        g_batch,
        u_batch,
        i_batch,
        genres,
        recommender,
        top_k,
        batch_size,
    ):

        gender_loss = 0
        bpr_loss = self.bpr_loss.compute(preds)
        f = g_batch == 1  # bool array is t when female
        m = g_batch == 0
        unique_items = torch.unique(i_batch)

        slice_ind = preds.shape[0] // 2
        pos_pred = preds[:slice_ind]
        r_batch = torch.ones_like(pos_pred)  # implicit feedback

        U_val = 0
        for i in unique_items:
            current_item_ind = i_batch == i
            female = f & current_item_ind
            male = m & current_item_ind

            E_g_yj = (
                pos_pred[female].mean() if female.any() else 0
            )  # female avg pred score
            E_mg_yj = pos_pred[male].mean() if male.any() else 0  # male avg pred score
            E_g_rj = (
                r_batch[female].mean() if female.any() else 0
            )  # female avg rating (actual)
            E_mg_rj = (
                r_batch[male].mean() if male.any() else 0
            )  # male avg rating (actual)
            U_val = U_val + abs((E_g_yj - E_mg_yj) - (E_g_rj - E_mg_rj))

        if self.a != 0:
            gender_loss = U_val / len(unique_items)
            loss = (
                self.a * gender_loss * (batch_size * max(bpr_loss))
                + (1 - self.a) * bpr_loss.sum()
            )

        else:
            loss = bpr_loss.sum()

        return loss


class BPR_loss_edit:
    def __init__(self):
        # super().__init__(reduction=reduction)
        self.name = "BPRLOSS"
        # self.a = a

    def bpr_loss(
        users_emb_final,
        users_emb_0,
        pos_items_emb_final,
        pos_items_emb_0,
        neg_items_emb_final,
        neg_items_emb_0,
        lambda_val,
    ):

        reg_loss = lambda_val * (
            users_emb_0.norm(2).pow(2)
            + pos_items_emb_0.norm(2).pow(2)
            + neg_items_emb_0.norm(2).pow(2)
        )

        pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
        pos_scores = torch.sum(pos_scores, dim=-1)

        neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
        neg_scores = torch.sum(neg_scores, dim=-1)

        loss = (
            -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores))
            + reg_loss
        )

        return loss

    def compute(self, preds):
        slice_ind = preds.shape[0] // 2
        pos_pred = preds[:slice_ind]
        neg_pred = preds[slice_ind:]

        score_diff = pos_pred - neg_pred
        fl = -score_diff.sigmoid().log()

        return fl
