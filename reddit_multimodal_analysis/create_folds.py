import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    df = pd.read_csv('../memotion_dataset_7k/labels.csv')
    df = df.drop('Unnamed: 0', axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    df['label'] = df['offensive']
    df['label'] = np.where(df['label'] == 'hateful_offensive', 'very_offensive', df['label'])
    
    mskf = StratifiedKFold(n_splits=5)

    df['kfold'] = -1
    for fold, (train, valid) in enumerate(mskf.split(X=df, y=df['label'])):
        df.loc[valid, 'kfold'] = fold
    
    df['label'] = df['label'].map({
        'not_offensive': 0, 
        'slight': 1, 
        'very_offensive': 2
    })

    df.to_csv('../memotion_dataset_7k/folds.csv', index=False)
