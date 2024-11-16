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
            user_batch = torch.cat((u_batch, u_batch), dim=0)

            # print(
            #     f"pos items {i_batch.shape} neg items {j_batch.shape} all {item_batch.shape}"
            # )
            # print(item_batch.shape)
            # print(user_batch.shape)

            # print(item_batch[:256])
            # print(item_batch[256:])
            # print("_" * 10)

            # print(item_batch2.shape)

            # r_batch = torch.tensor(r_batch, dtype=torch.float).to(device)

            preds = model(user_batch, item_batch)
            if _ == n_epochs:
                printLoss = True
            # loss = loss_func.compute(preds, batch_size, u_batch)
            loss = new_loss.forward(
                preds,
                # r_batch,
                genders,
                u_batch,
                i_batch,
                genress,
                # cat_batch,
                recommender,
                top_k,
                batch_size,
                # train_set.max_rating,
                # printLoss,
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
        # r_batch,
        g_batch,
        u_batch,
        i_batch,
        genres,
        recommender,
        top_k,
        batch_size,
        # max_rating,
        # printLoss=False,
    ):

        gender_loss = 0
        bpr_loss = self.bpr_loss.compute(preds)

        if self.a != 0:
            glmf = GenderLossMF(g_batch, u_batch, genres, recommender, top_k)
            gender_loss = glmf.compute()
            gender_loss = torch.sigmoid(0.1 * (gender_loss - 0.5))
            loss = (
                self.a * gender_loss * (batch_size * max(bpr_loss))
                + (1 - self.a) * bpr_loss.sum()
            )
            # print(
            #     f"bpr loss {bpr_loss.sum()} max(bpr) {max(bpr_loss)*batch_size} gloss={gender_loss*max(bpr_loss)*batch_size} loss ={loss}"
            # )

        else:
            loss = bpr_loss.sum()
            # print("*" * 10)
            # print(loss)
            # print("*" * 10)

        # glmf = GenderLossMF(
        #         g_batch, u_batch, i_batch, diff, genres, recommender, top_k
        #     )

        # print(
        #     f"{type(loss)} {type(mse_loss.sum())} { type(gender_loss * (batch_size * max(mse_loss)))}"
        # )
        # print(
        #     f"loss {loss}  bpr_loss {bpr.sum()} gloss {gender_loss * (batch_size * max(bpr))} pure gloss {gender_loss}"
        # )
        # print(gender_loss.requires_grad)
        # print(loss.requires_grad)
        # print(mse_loss)
        # print(gender_loss)

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
        # print("_" * 10)
        # print(
        #     f"preds shape {preds.shape} pos preds { pos_pred.shape} neg preds {neg_pred.shape}"
        # )
        # print("_" * 10)

        # self.pred = pred
        # self.ground_truth = ground_truth
        # pos_item = ground_truth == 1
        # neg_item = ground_truth == 0
        # pos = pred[pos_item]
        # neg = pred[neg_item]

        score_diff = pos_pred - neg_pred
        fl = -score_diff.sigmoid().log()

        return fl

    def compute_bpr_loss(self, pred, gt, uid):
        bpr_list = []
        bpr_user_list = []
        unique_users = torch.unique(uid)  # Get unique users

        for user in unique_users:
            bp_u = 0
            user_pos_scores = pred[(uid == user) & (gt == 1)]
            user_neg_scores = pred[(uid == user) & (gt == 0)]
            for pos_score in user_pos_scores:
                for neg_score in user_neg_scores:
                    score_diff = pos_score - neg_score
                    bploss = -score_diff.sigmoid().log()
                    bpr_list.append(bploss)
                    bp_u += bploss
            bpr_user_list.append(bp_u)
        # print(">" * 8)
        # print(bpr_user_list)
        # print(len(unique_users))
        # print(len(bpr_user_list))
        # print(torch.sum(torch.stack(bpr_user_list)))
        bpr_loss = torch.stack(bpr_list)
        return bpr_loss

    # # Data
    # uid = [1, 2, 3, 1, 2, 3, 1, 1]
    # gt = [1, 1, 1, 0, 0, 0, 1, 0]
    # pred = [0.3, 0.4, 0.2, 0, 0.33, 0.2, 0.1, 0]

    # # Calculate BPR Loss
    # bpr_loss = compute_bpr_loss(uid, gt, pred)
    # print("BPR Loss:", bpr_loss.item())

    # loss = -(pos - neg).sigmoid().log().sum()

    #         items_total = truth.shape[1]
    # nll = 0
    # for user, predictUser in zip(truth, predict):
    #     pos_idx = user.clone().detach()
    #     preUser = predictUser[pos_idx]
    #     non_zero_list = list(itertools.chain.from_iterable(torch.nonzero(user)))
    #     random_list = list(set(range(0, items_total)) - set(non_zero_list))
    #     random.shuffle(random_list)
    #     neg_idx = torch.tensor(random_list[: len(preUser)])
    #     score = preUser - predictUser[neg_idx]
    #     nll += -torch.mean(torch.nn.LogSigmoid()(score))
    # return nll
