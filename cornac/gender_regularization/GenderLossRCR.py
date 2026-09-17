import torch


class GenderLossRCR(object):
    def __init__(self, gender, users, genres, recommender, top_k):
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
        
        
        genre_sum = genres.sum(dim=-1, keepdim=True).to(device)
        genre_props = genres.to(device) / genre_sum.clamp(min=1.0)
        reco_prop_sum = reco_matrices @ genre_props
        train_prop_sum = genre_props.sum(dim=0).detach().clamp(min=1e-8)
        # user_dist = reco_genre/top_k
        
        user_dist = reco_prop_sum / train_prop_sum                        

        self.female_reco_dist = (
            user_dist[f].mean(dim=0) if f.any() else None
        )
        self.male_reco_dist = (
            user_dist[m].mean(dim=0) if m.any() else None
        )
        
    def compute(self):
        retVal_2 = torch.sum(torch.abs(self.male_reco_dist - self.female_reco_dist))
        # print(f" retVal_2{ retVal_2}")

        return retVal_2
