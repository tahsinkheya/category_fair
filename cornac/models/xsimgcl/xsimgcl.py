import torch.nn as nn
import dgl
import numpy as np
import scipy.sparse as sp
import dgl.function as fn
import torch.nn.functional as F
import torch
from ..lightgcn.lightgcn import construct_graph, GCNLayer

USER_KEY = "user"
ITEM_KEY = "item"


def info_nce(z1, z2, temperature):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(sim, labels)


# https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py


class Model(nn.Module):
    def __init__(
        self,
        g,
        in_size,
        eps,
        n_layers,
        layer_cl,
        lambda_reg,
        tau=0.2,
    ):
        # def __init__(self, data, emb_size, eps, n_layers, layer_cl): xsimgcl
        super().__init__()
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.eps = eps
        self.lambda_reg = lambda_reg
        self.tau = tau
        self.layers = nn.ModuleList([GCNLayer() for _ in range(self.n_layers)])
        self.initializer = nn.init.xavier_uniform_

        self.feature_dict = nn.ParameterDict(
            {
                ntype: nn.Parameter(
                    self.initializer(torch.empty(g.num_nodes(ntype), in_size))
                )
                for ntype in g.ntypes
            }
        )

    def forward(
        self,
        in_g,
        users=None,
        pos_items=None,
        neg_items=None,
        perturbed=False,
    ):
        h_dict = {ntype: self.feature_dict[ntype] for ntype in in_g.ntypes}
        all_user_embeddings = []
        all_item_embeddings = []

        user_cl_embeddings = None
        item_cl_embeddings = None
        for k, layer in enumerate(self.layers):
            h_dict = layer(in_g, h_dict)
            if perturbed:
                for ntype in [USER_KEY, ITEM_KEY]:
                    random_noise = torch.rand_like(h_dict[ntype])
                    h_dict[ntype] = (
                        h_dict[ntype]
                        + torch.sign(h_dict[ntype])
                        * F.normalize(
                            random_noise,
                            dim=-1,
                        )
                        * self.eps
                    )

            all_user_embeddings.append(h_dict[USER_KEY])
            all_item_embeddings.append(h_dict[ITEM_KEY])
            if k == self.layer_cl - 1:
                user_cl_embeddings = h_dict[USER_KEY]
                item_cl_embeddings = h_dict[ITEM_KEY]

        user_embeds = torch.stack(
            all_user_embeddings,
            dim=1,
        )

        item_embeds = torch.stack(
            all_item_embeddings,
            dim=1,
        )

        user_embeds = torch.mean(
            user_embeds,
            dim=1,
        )

        item_embeds = torch.mean(
            item_embeds,
            dim=1,
        )
        u_g_embeddings = user_embeds if users is None else user_embeds[users]

        pos_i_g_embeddings = (
            item_embeds if pos_items is None else item_embeds[pos_items]
        )

        neg_i_g_embeddings = (
            item_embeds if neg_items is None else item_embeds[neg_items]
        )

        if perturbed:

            return (
                u_g_embeddings,
                pos_i_g_embeddings,
                neg_i_g_embeddings,
                user_embeds,
                item_embeds,
                user_cl_embeddings,
                item_cl_embeddings,
            )

        return (
            u_g_embeddings,
            pos_i_g_embeddings,
            neg_i_g_embeddings,
            user_embeds,
            item_embeds,
        )

    def cl_loss(
        self,
        batch_users,
        batch_items,
        user_view1,
        user_view2,
        item_view1,
        item_view2,
    ):

        device = user_view1.device

        u_idx = torch.unique(
            torch.tensor(
                batch_users,
                dtype=torch.long,
                device=device,
            )
        )

        i_idx = torch.unique(
            torch.tensor(
                batch_items,
                dtype=torch.long,
                device=device,
            )
        )

        user_loss = info_nce(user_view1[u_idx], user_view2[u_idx], self.tau)

        item_loss = info_nce(item_view1[i_idx], item_view2[i_idx], self.tau)

        return user_loss + item_loss

    def loss_fn(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(1)
        neg_scores = (users * neg_items).sum(1)

        bpr_loss = F.softplus(neg_scores - pos_scores).mean()
        reg_loss = (
            (1 / 2)
            * (
                torch.norm(users) ** 2
                + torch.norm(pos_items) ** 2
                + torch.norm(neg_items) ** 2
            )
            / len(users)
        )

        # print("?>??>>?>?>?>")
        # print(bpr_loss)
        # print(F.softplus(neg_scores - pos_scores).shape)
        # print("?>??>>?>?>?>")

        return (
            bpr_loss + self.lambda_reg * reg_loss,
            bpr_loss,
            reg_loss,
            F.softplus(neg_scores - pos_scores),
        )
