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
        dropout,
        user_gender=None,
        item_cat=None,
    ):
        super(MF, self).__init__()

        self.use_bias = use_bias
        self.global_mean = global_mean
        self.dropout = nn.Dropout(p=dropout)

        self.u_factors = nn.Embedding(*u_factors.shape)
        self.i_factors = nn.Embedding(*i_factors.shape)
        self.u_factors.weight.data = torch.from_numpy(u_factors)
        self.i_factors.weight.data = torch.from_numpy(i_factors)
        self.user_gender = user_gender
        self.item_cat = item_cat

        if self.item_cat is not None:
            self.cat_embedding_size = 5

            self.i_cat = nn.Embedding(
                num_embeddings=self.item_cat.shape[1],
                embedding_dim=self.cat_embedding_size,
            )
            # self.i_cat.weight.data = torch.from_numpy(self.item_cat).float()

        if self.user_gender is not None:
            self.gender_embedding_size = 5
            self.u_genders = nn.Embedding(
                num_embeddings=2, embedding_dim=self.gender_embedding_size
            )
            # self.u_genders.weight.data = (
            #     torch.from_numpy(self.user_gender).float().view(-1, 1)
            # )  # reshape to ensure embedding dimension matches
            # self.u_genders.weight.requires_grad = False

        assert not torch.isnan(self.u_genders.weight.data).any()

        self.u_linear = nn.Linear(
            u_factors.shape[1] + self.gender_embedding_size,
            u_factors.shape[1],
        )
        self.i_linear = nn.Linear(
            i_factors.shape[1] + self.item_cat.shape[1] * self.cat_embedding_size,
            i_factors.shape[1],
        )

        if use_bias:
            self.u_biases = nn.Embedding(*u_biases.shape)
            self.i_biases = nn.Embedding(*i_biases.shape)
            self.u_biases.weight.data = torch.from_numpy(u_biases)
            self.i_biases.weight.data = torch.from_numpy(i_biases)

    def forward(self, uids, iids, genders, categories):
        ues = self.u_factors(uids)

        ibatch_items = categories[iids.numpy()]

        ies = self.i_factors(iids)
        cat_ies = self.i_cat(torch.tensor(ibatch_items))
        cat_ies = cat_ies.view(cat_ies.size(0), -1)

        assert not torch.isnan(uids).any()
        ubatch_genders = genders[uids.numpy()]
        ges = self.u_genders(torch.tensor(ubatch_genders))
        assert not torch.isnan(ges).any()

        if self.user_gender is not None:
            ues = torch.cat((ues, ges), dim=-1)
            ues = self.u_linear(ues)
        if self.item_cat is not None:
            ies = torch.cat((ies, cat_ies), dim=-1)
            ies = self.i_linear(ies)

        preds = (self.dropout(ues) * self.dropout(ies)).sum(dim=1, keepdim=True)
        if self.use_bias:
            preds += self.u_biases(uids) + self.i_biases(iids) + self.global_mean

        return preds.squeeze()


def find_nan_indices(tensor):
    nan_mask = torch.isnan(tensor)
    nan_indices = torch.nonzero(nan_mask, as_tuple=False)
    return nan_indices


def find_gender_loss(preds, genders, uids):
    ubatch_genders = genders[uids.numpy()]
    female = np.where(ubatch_genders == 1)[0]
    male = np.where(ubatch_genders == 0)[0]
    avg_f_pred = np.mean(preds.detach().numpy()[female])
    avg_m_pred = np.mean(preds.detach().numpy()[male])

    return (avg_f_pred - avg_m_pred) ** 2


def learn(
    model,
    train_set,
    n_epochs,
    batch_size=256,
    learning_rate=0.01,
    reg=0.0,
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

            preds = model(u_batch, i_batch, model.user_gender, model.item_cat)
            mse_loss = criteria(preds, r_batch)

            gender_loss = find_gender_loss(preds, model.user_gender, u_batch)
            loss = alpha * gender_loss + (1 - alpha) * mse_loss

            # print(
            #     f"Batch {batch_id}, MSE Loss: {mse_loss.item()}, Gender Loss: {gender_loss}, Combined Loss: {loss.item()}"
            # )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.item()
            count += len(u_batch)

            if batch_id % 10 == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))
