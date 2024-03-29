import keras
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras import optimizers
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Flatten, Dense, Input, Subtract, Multiply, Lambda, Dropout
from keras.layers.normalization import BatchNormalization
from keras.engine import Model
from scipy.misc import imread, imresize, imshow
from keras import backend as K
from keras.engine.topology import Layer
from keras.objectives import categorical_crossentropy
import random
import numpy as np
import tensorflow as tf
import gc
import os
import argparse
from generate_small_dataset import generate_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_data_from_file(file):
    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    data_list = []
    for i, val in enumerate(content):
        ii = val.split(' ')
        temp = [ii[0], ii[1], ii[2], ii[3], ii[4]]
        data_list.append(temp)
    data_list = np.asarray(data_list)
    return data_list


def load_data(training_np):
    training = get_data_from_file(training_np)
    size = training.shape[0]
    train_data = np.zeros((size, 224, 224, 6), dtype=np.float32)
    train_labels = np.zeros((size, 3))
    count = 0
    for i in training:
        if count >= size:
            break
        try:
            img1 = imread(base_dir + i[1])
            img1 = imresize(img1, (224, 224))
            img1 = np.float32(img1)

            if img1.ndim == 2:
                img1 = np.expand_dims(img1, axis=2)
                img1 = np.concatenate((img1, img1, img1), axis=-1)
            elif img1.shape[2] == 4:
                img1 = img1[:, :, :3]

            img1[:, :, 0] -= 93.5940
            img1[:, :, 1] -= 104.7624
            img1[:, :, 2] -= 129.1863

            train_data[count, :, :, 0:3] = img1
            # image 2
            img2 = imread(base_dir + i[3])
            img2 = imresize(img2, (224, 224))
            img2 = np.float32(img2)

            if img2.ndim == 2:
                img2 = np.expand_dims(img2, axis=2)
                img2 = np.concatenate((img2, img2, img2), axis=-1)
            elif img2.shape[2] == 4:
                img2 = img2[:, :, :3]

            train_data[count, :, :, 3:6] = img2

            img2[:, :, 0] -= 93.5940
            img2[:, :, 1] -= 104.7624
            img2[:, :, 2] -= 129.1863

            train_labels[count, 0] = int(i[0])
            train_labels[count, 1] = int(i[2])
            train_labels[count, 2] = int(i[4])

            count += 1
        except Exception:
            pass
    train_data /= 255.0
    return train_data, train_labels


class update_weights(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        # get weights
        lr = 0.0001
        lag_mul = 0.2
        W_weights = (self.model.layers[8].get_weights()[0])
        P_C_weights = (self.model.layers[9].get_weights()[0])
        P_V_weights = (self.model.layers[10].get_weights()[0])

        # # update Shared Layer W
        update_W = lr * lag_mul * np.dot(
            np.dot(P_C_weights, np.transpose(P_C_weights)),
            W_weights) + lr * lag_mul * np.dot(
                np.dot(P_V_weights, np.transpose(P_V_weights)), W_weights)
        W_weights = W_weights - (update_W)
        self.model.layers[8].set_weights(
            ((W_weights), self.model.layers[8].get_weights()[1]))

        # # update Unique Layer P_C
        update_P_C = lr * lag_mul * np.dot(
            np.dot(W_weights, np.transpose(W_weights)), P_C_weights)
        P_C_weights = P_C_weights - (update_P_C)

        self.model.layers[9].set_weights(
            ((P_C_weights), self.model.layers[9].get_weights()[1]))
        update_P_C = None
        P_C_weights = None
        for i in range(3):
            gc.collect()

        # update Unique Layer P_V
        update_P_V = lr * lag_mul * np.dot(
            np.dot(W_weights, np.transpose(W_weights)), P_V_weights)
        P_V_weights = P_V_weights - (update_P_V)

        self.model.layers[10].set_weights(
            ((P_V_weights), self.model.layers[10].get_weights()[1]))
        update_P_V = None
        P_V_weights = None
        update_W = None
        W_weights = None
        for i in range(3):
            gc.collect()


def model():

    # VGG model initialization with pretrained weights

    vgg_model_cari = VGGFace(include_top=True, input_shape=(224, 224, 3))
    last_layer_cari = vgg_model_cari.get_layer('pool5').output
    for i in vgg_model_cari.layers[0:7]:
        i.trainable = False
    custom_vgg_model_cari = Model(vgg_model_cari.input, last_layer_cari)

    vgg_model_visu = VGGFace(include_top=True, input_shape=(224, 224, 3))
    last_layer_visu = vgg_model_visu.get_layer('pool5').output
    for i in vgg_model_visu.layers[0:7]:
        i.trainable = False
    custom_vgg_model_visu = Model(vgg_model_visu.input, last_layer_visu)
    # Input of the siamese network : Caricature and Visual images

    caricature = Input(shape=(224, 224, 3), name='caricature')
    visual = Input(shape=(224, 224, 3), name='visual')
    # Get the ouput of the net for caricature and visual images
    caricature_net_out = custom_vgg_model_cari(caricature)
    caricature_net_out = Flatten()(caricature_net_out)
    visual_net_out = custom_vgg_model_visu(visual)
    visual_net_out = Flatten()(visual_net_out)

    # Merge the two networks by taking the transformation P_C, P_V[Unique transformations of visual & Caricature] and W [shared transformation]
    caricature_net_out = Dense(4096, activation="relu")(caricature_net_out)
    visual_net_out = Dense(4096, activation="relu")(visual_net_out)

    # Unique Layer - Caricature
    P_C_layer = Dense(2084, activation="relu", name="P_C_layer")
    P_C = P_C_layer(caricature_net_out)

    # Unique Layer - Visual
    P_V_layer = Dense(2084, activation="relu", name="P_V_layer")
    P_V = P_V_layer(visual_net_out)

    # Shared layers
    W = Dense(
        2084, activation="relu", name="W", kernel_initializer='glorot_uniform')
    W_C = W(caricature_net_out)
    W_V = W(visual_net_out)

    d = keras.layers.Concatenate(axis=-1)([W_C, W_V])
    d_1 = Dense(2048, activation="relu")(d)
    d_2 = Dense(1024, activation="sigmoid")(d_1)
    d_3 = Dense(2, activation="softmax", name='verification')(d_2)

    # Merge Unique and Shared layers for getting the feature descriptor of the image
    feature_caricature = keras.layers.Concatenate(axis=-1)([P_C, W_C])
    feature_visual = keras.layers.Concatenate(axis=-1)([P_V, W_V])

    # CARICATURE Classification Network - Dense layers

    fc1_c = Dense(2048, activation="relu")(feature_caricature)
    drop1_c = Dropout(0.6)(fc1_c)
    fc2_c = Dense(1024, activation="relu")(drop1_c)
    drop2_c = Dropout(0.6)(fc2_c)
    fc3_c = Dense(
        nb_class, activation="softmax",
        name='caricature_classification')(drop2_c)

    # VISUAL Classification Network - Dense layers

    fc1_v = Dense(2048, activation="relu")(feature_visual)
    drop1_v = Dropout(0.6)(fc1_v)
    fc2_v = Dense(1024, activation="relu")(drop1_v)
    drop2_v = Dropout(0.6)(fc2_v)
    fc3_v = Dense(
        nb_class, activation="softmax", name='visual_classification')(drop2_v)

    model = Model([caricature, visual], [d_3, fc3_c, fc3_v])

    return model


def train(model, training_file, model_file, validation_file, epoch):
    x_train, y_train = load_data(training_file)
    print("Read " + training_file + " successfully!")
    x_val, y_val = load_data(validation_file)
    print("Read " + validation_file + " successfully!")

    train_labels_cate_cari = to_categorical(
        y_train[:, 0], num_classes=nb_class)
    train_labels_cate_vis = to_categorical(y_train[:, 1], num_classes=nb_class)
    train_labels_verification = to_categorical(y_train[:, 2], num_classes=2)

    val_labels_cate_cari = to_categorical(y_val[:, 0], num_classes=nb_class)
    val_labels_cate_vis = to_categorical(y_val[:, 1], num_classes=nb_class)
    val_labels_verification = to_categorical(y_val[:, 2], num_classes=2)

    up_weights = update_weights()
    # loss = custom_loss
    if os.path.exists(model_file):
        model.load_weights(model_file)
    sgd = optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

    model.compile(
        loss=[
            categorical_crossentropy, categorical_crossentropy,
            categorical_crossentropy
        ],
        loss_weights=[33, 33, 33],
        optimizer=sgd,
        metrics=['accuracy'])

    model.fit(
        [x_train[:, :, :, 0:3], x_train[:, :, :, 3:6]], [
            train_labels_verification, train_labels_cate_cari,
            train_labels_cate_vis
        ],
        batch_size=20,
        epochs=epoch,
        verbose=1,
        shuffle=True,
        callbacks=[up_weights],
        validation_data=([x_val[:, :, :, 0:3], x_val[:, :, :, 3:6]], [
            val_labels_verification, val_labels_cate_cari, val_labels_cate_vis
        ]))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir', type=str, default='./af2019-ksyun-training-20190416',
    #                     help="Path to base directory.")
    # parser.add_argument('--nb_class', type=int, default=85,
    #                     help="The number of people.")
    # parser.add_argument('--training_file', type=str, default='./af2019-ksyun-training-20190416/training.txt',
    #                     help="Path to training file.")
    # parser.add_argument('--validation_file', type=str, default='./af2019-ksyun-training-20190416/validation.txt',
    #                     help='Path to validation file.')
    # parser.add_argument('--model', type=str, default='./model/hydxj_model2.h5',
    #                     help='Path to save model.')
    #
    # args = parser.parse_args()
    #
    # nb_class = args.nb_class  # custom parameters
    # base_dir = args.base_dir  # Change it accordingly!
    #
    # t_model = model()
    # train(t_model, args.training_file, args.model, args.validation_file)
    # if not os.path.isdir('./model'):
    #     os.mkdir('./model')
    # t_model.save_weights(args.model)

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dataset', type=str, default='./af2019-ksyun-training-20190416',
                        help="Path to training dataset.")
    parser.add_argument('--nb_class', type=int, default=85,
                        help="The number of trainging people.")
    parser.add_argument('--model', type=str, default='./model/hydxj_model3.h5',
                        help='Path to save model.')
    parser.add_argument('--epoch', type=int, default=2,
                        help="train epoch")

    args = parser.parse_args()

    nb_class = args.nb_class  # custom parameters
    base_dir = args.training_dataset  # Change it accordingly!

    generate_data(args.training_dataset, nb_class)

    t_model = model()
    train(t_model, base_dir + '/training.txt', args.model, base_dir + '/validation.txt', args.epoch)
    if not os.path.isdir('./model'):
        os.mkdir('./model')
    t_model.save_weights(args.model)
