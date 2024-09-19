import numpy as np
import proc_img as lg


def data_generator(img_paths, labels, batch_size):
    while True:
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            images = [lg.proc_img(path) for path in batch_paths]
            images = np.array(images)

            yield images, np.array(batch_labels)
