import torch


class GenderLossMF(object):
    def __init__(self, gender, users, genres, recommender, top_k):
        unique_users = torch.unique(users)
        # print(gender)

        gender_mask = torch.zeros_like(gender, dtype=torch.bool)
        unique_users_gender = gender[unique_users]
        gender_mask[unique_users] = True
        unique_users_gender2 = gender[gender_mask]
        # print(unique_users_gender.requires_grad)
        # print(unique_users_gender2.requires_grad)

        f = unique_users_gender == 1
        m = unique_users_gender == 0
        recommendations_t = torch.stack(
            [
                recommender.differentiable_rank(u, k=top_k)[0][:top_k]
                for u in unique_users
            ]
        )
        # print(")" * 10)
        # print(genres.requires_grad)
        # print(")" * 10)
        # print("--" * 20)
        # print(recommendations_t.requires_grad)
        # print("--" * 20)

        # print(",,,;,;,;,;,;,;,;,;,;,")

        # print(recommender.differentiable_rank(19, k=10)[0][:10])
        # print(",,,;,;,;,;,;,;,;,;,;,")

        # try:
        #     print(",,,;,;,;,;,;,;,;,;,;,")
        #     print(recommender.rank(19, k=10)[0][:10])
        #     print(recommender.differentiable_rank(19, k=10)[0][:10])
        #     print(",,,;,;,;,;,;,;,;,;,;,")
        # except Exception as e:
        #     print("heyo")
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
        reco_movie_tensor = genres[recommendations_t]
        # print("-" * 20)
        # print(reco_movie_tensor.requires_grad)
        # print("-" * 20)
        sum_genres = reco_movie_tensor.sum(axis=-1, keepdim=True)
        reco_movie_tensor_prop = torch.where(
            sum_genres == 0, torch.tensor(0.0), reco_movie_tensor / sum_genres
        )
        reco_movie_tensor_prop_mean = reco_movie_tensor_prop.mean(
            dim=1
        )  # genre dist per user

        self.male_reco_dist = reco_movie_tensor_prop_mean[m].mean(dim=0)
        self.female_reco_dist = reco_movie_tensor_prop_mean[f].mean(dim=0)
        # print("--" * 20)
        # print(reco_movie_tensor.requires_grad)
        # print(sum_genres.requires_grad)
        # print("--" * 20)

    def compute(self):
        retVal_2 = torch.sum(torch.abs(self.male_reco_dist - self.female_reco_dist))

        return retVal_2
