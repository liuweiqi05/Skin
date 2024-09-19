import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def proc_img(path_img, img_h=180, img_w=180):
    # images = []
    # for i in df_imgs["img_path"]:
    #     img = load_img(i, target_size=(img_h, img_w))
    #     img = img_to_array(img)
    #     images.append(img)
    # images = np.array(images)
    # return images
    img = load_img(path_img, target_size=(img_h, img_w))
    img = img_to_array(img)
    return img

