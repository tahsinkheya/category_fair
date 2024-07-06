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
            self.i_cat = nn.Embedding(self.item_cat.shape[0], self.item_cat.shape[1])
            self.i_cat.weight.data = torch.from_numpy(self.item_cat).float()

        if self.user_gender is not None:
            self.gender_embedding_size = 1
            self.u_genders = nn.Embedding(
                self.user_gender.shape[0], self.gender_embedding_size
            )
            self.u_genders.weight.data = (
                torch.from_numpy(self.user_gender).float().view(-1, 1)
            )  # reshape to ensure embedding dimension matches
        assert not torch.isnan(self.u_genders.weight.data).any()

        self.u_linear = nn.Linear(
            u_factors.shape[1] + self.gender_embedding_size,
            u_factors.shape[1],
        )
        self.i_linear = nn.Linear(
            i_factors.shape[1] + self.item_cat.shape[1],
            i_factors.shape[1],
        )

        if use_bias:
            self.u_biases = nn.Embedding(*u_biases.shape)
            self.i_biases = nn.Embedding(*i_biases.shape)
            self.u_biases.weight.data = torch.from_numpy(u_biases)
            self.i_biases.weight.data = torch.from_numpy(i_biases)

    def forward(self, uids, iids):
        ues = self.u_factors(uids)
        ies = self.i_factors(iids)
        cat_ies = self.i_cat(iids)
        assert not torch.isnan(uids).any()
        ges = self.u_genders(uids)
        assert not torch.isnan(ges).any()

        if self.user_gender is not None:
            ues = torch.cat((ues, ges), dim=-1)
            ues = self.u_linear(ues)
        if self.item_cat is not None:
            ies = torch.cat((ies, cat_ies), dim=-1)
            ies = self.i_linear(ies)

        if torch.isnan(ues).any():
            print(find_nan_indices(ues))
            raise ValueError("NaNs found in embeddings before USER concatenation")

        if torch.isnan(ies).any():
            raise ValueError("NaNs found in embeddings ITEM before concatenation")

        preds = (self.dropout(ues) * self.dropout(ies)).sum(dim=1, keepdim=True)
        if self.use_bias:
            preds += self.u_biases(uids) + self.i_biases(iids) + self.global_mean

        return preds.squeeze()


def find_nan_indices(tensor):
    nan_mask = torch.isnan(tensor)
    nan_indices = torch.nonzero(nan_mask, as_tuple=False)
    return nan_indices


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
):
    model = model.to(device)
    criteria = nn.MSELoss(reduction="sum")
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

            preds = model(u_batch, i_batch)
            loss = criteria(preds, r_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.item()
            count += len(u_batch)

            if batch_id % 10 == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))
