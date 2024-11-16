import cornac
from cornac.data import Reader
from cornac.datasets import movielens
from cornac.data import Dataset, FeatureModality
from cornac.eval_methods import RatioSplit, StratifiedSplit
from cornac.metrics import RMSE, AUC, NDCG, Precision, Recall
from cornac.models import (
    MF,
    ItemKNN,
    UserKNN,
    NMF,
    BPR,
    LightGCN,
    SVD,
    MostPop,
    VAECF,
    NeuMF,
)
import pandas as pd
import numpy as np
import random
import math
from collections import OrderedDict

# import seaborn as sns
# import matplotlib.pyplot as plt

reader = Reader()
rating_data_pd = pd.read_csv(
    "./cornac/data_c/indexed_interactions.csv",
    sep="\t",
    header=None,
    names=["userID", "itemID", "Rating", "Timestamp"],
)
rating_data = rating_data_pd.to_numpy()
rating_data.__len__()
rating_data_pd


movies = pd.read_csv("./cornac/data_c/indexed_movies.csv")

movies = movies.drop(columns=movies.columns[0])
movies[:4]

unique_genres = [
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
for genre in unique_genres:
    movies[genre] = 0
for index, row in movies.iterrows():
    genres = row["genres"].split("|")
    for genre in genres:
        movies.at[index, genre] = 1

genre = movies[unique_genres]
item_features_numpy = genre.to_numpy()
item_features = {
    str(item_id): {"genre_" + str(idx): value for idx, value in enumerate(row)}
    for item_id, row in enumerate(item_features_numpy)
}
ids = list(range(0, 3416))

users = pd.read_csv("./cornac/data_c/u_id_mapping.csv", sep="\t")
users = users.drop(columns=users.columns[0])
gender_map = {"M": 0, "F": 1}
users["Gender"] = users["Gender"].map(gender_map)
user_features_numpy = users.to_numpy()
print(user_features_numpy.shape)
print(item_features_numpy.shape)
dataset = rating_data
unique_genres.__len__()

movies = movies.sort_values(by="itemID")
movies
users
user_features_numpy[:, 1]
dataset = rating_data
unique_genres.__len__()
ratio_split = StratifiedSplit(
    data=dataset,
    test_size=0.2,
    rating_threshold=0,
    val_size=0.1,
    seed=123,
    verbose=True,
    chrono=True,
    user_features=user_features_numpy[:, 0],
    item_features=item_features_numpy,
    exclude_unknowns=False,
)
rec_50 = cornac.metrics.Recall(k=50)
ndcg_50 = cornac.metrics.NDCG(k=50)
auc = cornac.metrics.AUC()
rmse = cornac.metrics.RMSE()
prec = cornac.metrics.Precision(k=50)
hr = cornac.metrics.HitRatio(k=50)
mrr = cornac.metrics.MRR()
map = cornac.metrics.MAP()
f1 = cornac.metrics.FMeasure(k=50)

models = []

alpha_values = [0.4, 0.5, 0.6]

# alpha_values =[0]
for i in range(len(alpha_values)):
    # learning
    model = MF(
        k=40,
        seed=123,
        name=f"a={alpha_values[i]} mf",
        backend="pytorch",
        verbose=True,
        optimizer="adam",
        batch_size=256,
        alpha=alpha_values[i],
        learning_rate=0.001,
        top_k=50,
        max_iter=11,
        run_mode="bce",
        # early_stopping=True
    )
    models.append(model)


# models = [model_1, model_2, model_3, model_4, model_5]
cornac.Experiment(
    ratio_split,
    models=models,
    metrics=[rec_50, ndcg_50, auc, rmse, prec, hr, mrr, map, f1],
).run()


# Early stopping:
# - best epoch = 1, stopped epoch = 11
# - best monitored value = 0.067115 (delta = -0.000222)

user_ids = users.to_numpy()[:, 1]
item_ids = movies.to_numpy()[:, 2]
item_ids.__len__()
# get the top_k ratings for all users:
top_k = 50
reco_matrix = np.zeros((len(models), len(user_ids), top_k), dtype=int)
reco_matrix_mapped_items = np.zeros(
    (len(models), len(user_ids), len(item_ids)), dtype=int
)
reco_matrix_mapped_scores = np.zeros(
    (len(models), len(user_ids), len(item_ids)), dtype=float
)
reco_matrix_all = np.zeros((len(models), len(user_ids), len(item_ids)), dtype=int)


for u in user_ids:
    for i in range(len(models)):
        reco_items = models[i].recommend(u)
        items_mapped, mapped_scores = models[i].rank(
            user_idx=u, item_indices=list(item_ids)
        )
        reco_matrix_mapped_items[i][u] = items_mapped
        reco_matrix_mapped_scores[i][u] = mapped_scores
        reco_matrix_all[i][u] = reco_items
        reco_matrix[i][u] = reco_items[:top_k]


np.save("reco_matrix_mf_1m_456_bs256e11k40bpr.npy", reco_matrix)
np.save("reco_matrix_all_mf_1m_456_bs256e11k40bpr.npy", reco_matrix_all)

import pickle

with open("mf1m_456_bs256e11k40bpr.pkl", "wb") as f:
    pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)
