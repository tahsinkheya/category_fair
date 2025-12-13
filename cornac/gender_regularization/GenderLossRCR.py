import torch


class GenderLossMFRCR(object):
    def __init__(self, gender, users, genres, recommender, top_k):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        sum_genres = reco_movie_tensor.sum(axis=-1, keepdim=True).to(device)
        reco_movie_tensor_prop = torch.where(
            sum_genres == 0, torch.tensor(0.0), reco_movie_tensor / sum_genres
        ).to(device)

        sum_genres_train = genres.sum(axis=-1, keepdim=True).to(device)
        train_movie_tensor_prop = torch.where(
            sum_genres_train == 0, torch.tensor(0.0), genres / sum_genres_train
        ).to(device)

        reco_movie_tensor_prop_mean = reco_movie_tensor_prop.mean(dim=1).to(
            device
        )  # genre dist per user for top 50
        reco_movie_tensor_prop_sum = reco_movie_tensor_prop.sum(dim=1).to(
            device
        )  # genre dist per user for top 50
        train_movie_tensor_prop_sum = train_movie_tensor_prop.sum(dim=0).to(device)

        reco_train_prop = reco_movie_tensor_prop_sum / train_movie_tensor_prop_sum
        # print(f"final output {reco_train_prop}")

        self.male_reco_dist = reco_train_prop[m].mean(dim=0)
        self.female_reco_dist = reco_train_prop[f].mean(dim=0)
        # print("yoyo")
        # print(f" self.male_reco_dist { self.male_reco_dist}")
        # print(f" self.female_reco_dist { self.female_reco_dist}")
        # print("--" * 20)
        # print(reco_movie_tensor.requires_grad)
        # print(sum_genres.requires_grad)
        # print("--" * 20)

    def compute(self):
        retVal_2 = torch.sum(torch.abs(self.male_reco_dist - self.female_reco_dist))
        # print(f" retVal_2{ retVal_2}")

        return retVal_2
