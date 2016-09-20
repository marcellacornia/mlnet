from __future__ import division
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os, cv2, sys
import numpy as np
from config import *
from utilities import preprocess_images, preprocess_maps, postprocess_predictions
from model import ml_net_model, loss


def generator(b_s, phase_gen='train'):
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith('.jpg')]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith('.jpg')]
    elif phase_gen == 'val':
        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith('.jpg')]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith('.jpg')]
    else:
        raise NotImplementedError

    images.sort()
    maps.sort()

    counter = 0
    while True:
        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c), preprocess_maps(maps[counter:counter + b_s], shape_r_gt, shape_c_gt)
        counter = (counter + b_s) % len(images)


def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith('.jpg')]
    images.sort()

    counter = 0
    while True:
        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c)
        counter = (counter + b_s) % len(images)


if __name__ == '__main__':
    phase = sys.argv[1]

    model = ml_net_model(img_cols=shape_c, img_rows=shape_r, downsampling_factor_product=10)
    sgd = SGD(lr=1e-3, decay=0.0005, momentum=0.9, nesterov=True)
    print("Compile ML-Net Model")
    model.compile(sgd, loss)

    if phase == 'train':
        print("Training ML-Net")
        model.fit_generator(generator(b_s=b_s), nb_imgs_train, nb_epoch=nb_epoch,
                            validation_data=generator(b_s=b_s, phase_gen='val'), nb_val_samples=nb_imgs_val,
                            callbacks=[EarlyStopping(patience=5),
                                       ModelCheckpoint('weights.mlnet.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True)])

    elif phase == "test":
        # path of output folder
        output_folder = ''

        if len(sys.argv) < 2:
            raise SyntaxError
        imgs_test_path = sys.argv[2]

        file_names = [f for f in os.listdir(imgs_test_path) if f.endswith('.jpg')]
        nb_imgs_test = len(file_names)

        print("Load weights ML-Net")
        model.load_weights('mlnet_salicon_weights.pkl')

        print("Predict saliency maps for " + imgs_test_path)
        predictions = model.predict_generator(generator_test(b_s=1, imgs_test_path=imgs_test_path), nb_imgs_test)

        for pred, name in zip(predictions, file_names):
            original_image = cv2.imread(imgs_test_path + name, 0)
            res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
            cv2.imwrite(output_folder + '%s' % name, res.astype(int))

    else:
        raise NotImplementedError
