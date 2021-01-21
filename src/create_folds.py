import pandas as pd
import os
from sklearn import model_selection

import config

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(config.INPUT_PATH, 'train.csv'))

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df["Purchase"].values

    kf = model_selection.KFold(n_splits=5)

    for fold, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = fold

    df.to_csv(os.path.join(config.INPUT_PATH, 'train_folds.csv'), index=False)