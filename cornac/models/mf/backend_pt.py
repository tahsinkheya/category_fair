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
from cornac.models.mf.GenderLossMF import GenderLossMF
from cornac.models.mf.GenderLossMF2 import GenderLossMF2

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
    new_loss = GenderMseLoss(a=alpha, reduction="sum")
    printLoss = False
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
                printLoss,
            )
            # loss = new_loss(
            #     preds, r_batch, g_batch, u_batch, i_batch, cat_batch, recommender, top_k
            # )
            # print(u_batch.requires_grad, r_batch.requires_grad, g_batch.requires_grad)

            # loss.retain_grad()
            # print(gender_loss.grad)
            # print(loss.grad)
            # print(mse_loss.grad)
            # print(type(loss))
            # print(type(mse_loss))
            # print(mse_loss.requires_grad, gender_loss.requires_grad, diff.requires_grad)

            # print(type(gender_loss))
            # print(type(mse_loss))
            # print(type(loss))

            optimizer.zero_grad()
            # loss.retain_grad()
            loss.backward()

            # # total_grad_norm = 0.0
            # # for name, param in model.named_parameters():
            # #     if param.grad is not None:
            # #         grad_norm = param.grad.norm(2).item()
            # #         total_grad_norm += grad_norm
            # #         # Log the gradient norm for each parameter
            # #         if batch_id % 10 == 0:
            # #             print(
            # #                 f"Epoch {_}, Batch {batch_id}, Grad Norm [{name}]: {grad_norm}"
            # #             )
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Grad Norm [{name}]: {param.grad.norm()}")
            #     else:
            #         print(f"No gradient for {name}")

            optimizer.step()

            sum_loss += loss.data.item()
            count += len(u_batch)

            if batch_id % 10 == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))


class GenderMseLoss(nn.MSELoss):
    def __init__(self, a, reduction):
        super().__init__(reduction=reduction)
        self.a = a
        # self.max_mse_loss = -1
        # self.max_gender_loss = -1
        # self.min_mse_loss = 10000
        # self.min_gender_loss = 10000
        self.max_mse_loss = 404.9754333496094
        self.max_gender_loss = 0.7370653866116046

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
        printLoss=False,
    ):
        mse_loss = super().forward(preds, r_batch)
        # print(mse_loss)
        f = g_batch == 1
        m = g_batch == 0
        diff = torch.abs(r_batch - preds)

        # equation 1 start____________________________
        # avg_f = diff[f].mean()
        # avg_m = diff[m].mean()
        # gender_loss_1 = torch.abs(avg_f - avg_m)
        # equation 1 end____________________________

        # equation 2 start____________________________
        # female = diff[f]
        # male = diff[m]
        # total_num_preds = r_batch.shape[0]
        # female_acc_count = (female < 0.5).sum().item()
        # male_acc_count = (male < 0.5).sum().item()

        # gender_loss_2 = abs(
        #     female_acc_count / total_num_preds - male_acc_count / total_num_preds
        # )
        # equation 2 end____________________________

        # equation 3 start____________________________
        gender_loss = 0
        if self.a != 0:
            glmf = GenderLossMF2(
                g_batch, u_batch, i_batch, diff, genres, recommender, top_k
            )
            gender_loss = glmf.compute()
            # loss = self.a * (gender_loss / self.max_gender_loss) + (1 - self.a) * (
            #     mse_loss / self.max_mse_loss
            # )
            loss = self.a * gender_loss + mse_loss

        else:
            loss = mse_loss

        # glmf = GenderLossMF(
        #         g_batch, u_batch, i_batch, diff, genres, recommender, top_k
        #     )
        # loss = glmf.compute()
        # loss.requires_grad = True

        # print(f"{type(loss)} {type(mse_loss)} { type(gender_loss)}")
        # print(mse_loss)

        # equation 3 end____________________________

        # need these for normalization____________
        # if loss > self.max_gender_loss:
        #     self.max_gender_loss = loss
        # if mse_loss > self.max_mse_loss:
        #     self.max_mse_loss = mse_loss
        # if printLoss:
        # print(f"mse_loss maximum {self.max_mse_loss}")
        # print(f"gender maximum {self.max_gender_loss}")
        # if gender_loss < self.min_gender_loss:
        #     self.min_gender_loss = gender_loss
        # if mse_loss < self.min_mse_loss:
        #     self.min_mse_loss = mse_loss
        # if printLoss:
        #     print(f"mse_loss min {self.min_mse_loss}")
        #     print(f"gender min {self.min_gender_loss}")
        # need these for normalization____________

        # print(f"gl{gender_loss*self.a} loss{loss} mseloss {mse_loss}")

        # loss = self.a * (gender_loss / self.max_gender_loss) + (1 - self.a) * (
        #     mse_loss / self.max_mse_loss
        # )

        print(f"gl{gender_loss} loss{loss} mseloss {mse_loss}")
        # print(gender_loss)
        # print(self.max_gender_loss)

        return loss
