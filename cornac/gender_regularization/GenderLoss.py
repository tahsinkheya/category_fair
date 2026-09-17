import torch


class GenderLoss(object):
    def __init__(self, gender, users, genres, recommender, top_k):
        # self.epoch = epoch
        # print(f"epoch {epoch}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        unique_users = torch.unique(users).to(device)
        # print(gender)

        gender_mask = torch.zeros_like(gender, dtype=torch.bool, device=device)
        unique_users_gender = gender[unique_users]
        gender_mask[unique_users] = True

        f = unique_users_gender == 1
        m = unique_users_gender == 0
        reco_matrices = []
        for u in unique_users:
            P, _, _ = recommender.differentiable_rank(u, k=top_k)
            reco_matrices.append(P)
        reco_matrices = torch.stack(reco_matrices, dim=0).to(device)
        
        # import pdb; pdb.set_trace()
        genre_sum = genres.sum(dim=-1, keepdim=True).to(device)
        genre_props = genres.to(device) / genre_sum.clamp(min=1.0)
        reco_genre = reco_matrices @ genre_props
        user_dist = reco_genre/top_k

        self.female_reco_dist = (
            user_dist[f].mean(dim=0)
            if f.any()
            else torch.zeros_like(user_dist[0])
        )
        self.male_reco_dist  = (
            user_dist[m].mean(dim=0)
            if m.any()
            else torch.zeros_like(user_dist[0])
        )

    def compute(self):

        retVal_2 = torch.sum(torch.abs(self.male_reco_dist - self.female_reco_dist))

        return retVal_2
