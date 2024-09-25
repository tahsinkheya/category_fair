import torch


class GenderLossMF(object):
    def __init__(self, gender, users, genres, recommender, top_k):
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        unique_users = torch.unique(users).to(device)
        # print(gender)

        gender_mask = torch.zeros_like(gender, dtype=torch.bool, device=device)
        unique_users_gender = gender[unique_users]
        gender_mask[unique_users] = True
        # unique_users_gender2 = gender[gender_mask]
        # print(unique_users_gender.requires_grad)
        # print(unique_users_gender2.requires_grad)

        f = unique_users_gender == 1
        m = unique_users_gender == 0
        recommendations_t = torch.stack(
            [
                recommender.differentiable_rank(u, k=top_k)[0][:top_k]
                for u in unique_users
            ]
        ).to(device)
     
        self.unique_genres = [
            "Action",
            "Thriller",
            "Romance",
            "Western",
            "Children's",
            "Mystery",
            "Fantasy",
            "Film-Noir",
            "Documentary",
            "Comedy",
            "Adventure",
            "Sci-Fi",
            "Horror",
            "Crime",
            "Musical",
            "War",
            "Animation",
            "Drama",
        ]
        reco_movie_tensor = genres[recommendations_t].to(device)
        # print("-" * 20)
        # print(reco_movie_tensor.requires_grad)
        # print("-" * 20)
        sum_genres = reco_movie_tensor.sum(axis=-1, keepdim=True).to(device)
        reco_movie_tensor_prop = torch.where(
            sum_genres == 0, torch.tensor(0.0), reco_movie_tensor / sum_genres
        ).to(device)
        reco_movie_tensor_prop_mean = reco_movie_tensor_prop.mean(
            dim=1
        ).to(device)  # genre dist per user

        self.male_reco_dist = reco_movie_tensor_prop_mean[m].mean(dim=0)
        self.female_reco_dist = reco_movie_tensor_prop_mean[f].mean(dim=0)
        # print("--" * 20)
        # print(reco_movie_tensor.requires_grad)
        # print(sum_genres.requires_grad)
        # print("--" * 20)

    def compute(self):
        retVal_2 = torch.sum(torch.abs(self.male_reco_dist - self.female_reco_dist))

        return retVal_2
