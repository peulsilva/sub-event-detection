from src.path_reader import BASE_PATH_TRAIN
from src.preprocessing import preprocess_data
from tqdm import tqdm
import pandas as pd
import os

TRAIN_IDX = [0, 2, 4, 7, 8, 11, 13, 14, 18, 19]
TEST_IDX = [1, 3, 5, 10, 12, 17]


def train_test_split():
    all_dfs = []

    for file in tqdm(os.listdir(BASE_PATH_TRAIN)):
        file_path = os.path.join(BASE_PATH_TRAIN, file)
        all_dfs.append(pd.read_csv(file_path))

    all_df = pd.concat(all_dfs)

    all_df = preprocess_data(all_df)

    train_df = {key: group.sort_values(by = "Timestamp") for key, group in all_df.query(f"MatchID in {TRAIN_IDX}").groupby("MatchID")}
    test_df = {key: group.sort_values(by = "Timestamp") for key, group in all_df.query(f"MatchID in {TEST_IDX}").groupby("MatchID")}

    return train_df, test_df
