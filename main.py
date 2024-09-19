import numpy as np
import model as ml
import proc_f as pf
import proc_img as lg
import dg_gen as dg
# from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

path_meta = "D:/Skin/train-metadata.csv"
path_img = "D:/Skin/train-image/image"
batch_size = 32
num_classes = 2

df = pf.proc_meta(path_meta)
img_paths, labels = pf.proc_img_lab(path_img, df)
X_train, X_test, y_train, y_test = train_test_split(img_paths, labels, test_size=0.2, random_state=42)
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)
train_g = dg.data_generator(X_train, y_train, batch_size)
val_g = dg.data_generator(X_test, y_test, batch_size)
train_steps = len(X_train) // batch_size
val_steps = len(X_test) // batch_size
# print(image_paths[:3], labels[:3])
# img = lg.proc_img(df_imgs)
# print(img.shape, img)
# images = []
# for i in df_imgs["img_path"]:
#     images.append(lg.proc_img(i))
# images = np.array(images)
#

model = ml.init_model()
history = model.fit(train_g, epochs=10, steps_per_epoch=train_steps, validation_data=val_g, validation_steps=val_steps)
val_loss, val_acc = model.evaluate(val_g, steps=val_steps)

print(f'Validation Accuracy: {val_acc:.4f}')
