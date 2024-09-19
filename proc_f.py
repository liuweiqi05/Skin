import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def proc_meta(path_meta):
    df = pd.read_csv(path_meta, low_memory=False)
    # df = shuffle(df)
    df['iddx_full'] = df['iddx_full'].map({'malignant': 1, 'benign': 0})
    df['iddx_full'].fillna(0, inplace=True)
    return df


def proc_img_lab(path_img, df):
    image_paths = []
    labels = []
    data = []
    for index, row in df.iterrows():
        image_id = row['isic_id']
        label = row['target']
        image_path = os.path.join(path_img, f"{image_id}.jpg")

        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(label)
            data.append([image_path, label])

    labels = np.array(labels)
    # df_imgs = pd.DataFrame(data, columns=['img_path', 'label'])

    return image_paths, labels
