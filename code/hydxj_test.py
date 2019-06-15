import keras
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras import optimizers
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.layers import Flatten, Dense, Input, Merge, Subtract, Multiply, Lambda, Dropout
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
import argparse
import csv
import os



def get_data_from_file(file, test_mode):
    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    data_list = []
    if test_mode == 1:
        for i, val in enumerate(content):
            ii = val.split(' ')
            temp = [ii[0], ii[1], ii[2], ii[3], ii[4]]
            data_list.append(temp)
    else:
        for i, val in enumerate(content):
            ii = val.split(' ')
            temp = [ii[0], ii[1], ii[2], ii[3]]
            data_list.append(temp)
    data_list = np.asarray(data_list)
    return data_list


def load_data(testing_np, test_mode):
    testing = get_data_from_file(testing_np, test_mode)
    size = testing.shape[0]
    test_data = np.zeros((size, 224, 224, 6), dtype=np.float32)
    test_labels = np.zeros(size)
    count = 0
    for i in testing:
        if count >= size:
            break
        img1 = imread(base_dir + i[1])
        img1 = imresize(img1, (224, 224))
        img1 = np.float32(img1)
        # print "img1:" + base_dir + i[1]
        # print img1.shape
        if img1.ndim == 2:
            img1 = np.expand_dims(img1, axis=2)
            img1 = np.concatenate((img1, img1, img1), axis=-1)
        elif img1.shape[2] == 4:
            img1 = img1[:, :, :3]

        img1[:, :, 0] -= 93.5940
        img1[:, :, 1] -= 104.7624
        img1[:, :, 2] -= 129.1863

        test_data[count, :, :, 0:3] = img1
        # image 2
        img2 = imread(base_dir + i[3])
        img2 = imresize(img2, (224, 224))
        img2 = np.float32(img2)
        # print "img2:" + base_dir + i[3]
        # print img2.shape
        if img2.ndim == 2:
            img2 = np.expand_dims(img2, axis=2)
            img2 = np.concatenate((img2, img2, img2), axis=-1)
        elif img2.shape[2] == 4:
            img2 = img2[:, :, :3]

        img2[:, :, 0] -= 93.5940
        img2[:, :, 1] -= 104.7624
        img2[:, :, 2] -= 129.1863

        test_data[count, :, :, 3:6] = img2

        if test_mode == 1:
            test_labels[count] = int(i[4])

        count += 1
    test_data /= 255.0
    if test_mode == 1:
        return test_data, test_labels
    else:
        return test_data


def write_csv_list(list_file, data, header = list()):
    if not os.path.isdir('./result'):
        os.mkdir('./result')

    with open(list_file, 'wb') as fh:
        csv_writer = csv.writer(fh)
        if len(header):
            csv_writer.writerow(header)
        for row in data:
            csv_writer.writerow(row)


def read_csv(path):
    csv_path = path + '/list.csv'
    csv_file = csv.reader(open(csv_path, 'r'))
    print("start read csv...")
    # store the path of cartoon pictures and people pictures, respectively
    with open(path + '/test.txt', 'w') as f:
        for pic in csv_file:
            img1 = pic[1]
            img2 = pic[2]
            if img1 == 'image1':
                continue
            pic_path = '0 ' + img1 + ' 0 ' + img2 + '\n'
            f.write(pic_path)
        print('read csv ok!')
    return path + '/test.txt'

def model():

    # VGG model initialization with pretrained weights

    vgg_model_cari = VGGFace(
        include_top=True, weights=None, input_shape=(224, 224, 3))
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

    # Unique layers
    P_C_layer = Dense(2084, activation="relu", name="P_C_layer")
    P_C = P_C_layer(caricature_net_out)

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

    # d = keras.layers.merge([W_C, W_V], mode=euc_dist, output_shape=euc_dist_shape, name='contrastive_loss')

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
    #
    # # VISUAL Classification Network - Dense layers
    #
    fc1_v = Dense(2048, activation="relu")(feature_visual)
    drop1_v = Dropout(0.6)(fc1_v)
    fc2_v = Dense(1024, activation="relu")(drop1_v)
    drop2_v = Dropout(0.6)(fc2_v)
    fc3_v = Dense(
        nb_class, activation="softmax", name='visual_classification')(drop2_v)

    model = Model([caricature, visual], [d_3, fc3_c, fc3_v])

    return model


def test(model, dataset_file, model_file, predition_file, test_mode):
    if test_mode == 1:
        x_test, y_test = load_data(dataset_file, test_mode)
    else:
        x_test = load_data(dataset_file, test_mode)

    model.load_weights(model_file)

    count = 0
    pred = model.predict(
        [x_test[:, :, :, 0:3], x_test[:, :, :, 3:6]], verbose=1)
    results = list()
    for i in range(0, pred[0].shape[0]):
        result = [i, pred[0][i][1]]
        results.append(result)
        if test_mode == 1:
            argmax_ver = np.argmax(pred[0][i])
            if argmax_ver == y_test[i]:
                count += 1
    if test_mode == 1:
        accur = count * 1.0 / pred[0].shape[0]
        print("Accur:" + str(accur))


    write_csv_list(predition_file, results, header=['group_id', 'confidence'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', type=str, default='./af2019-ksyun-testB-20190424',
                        help="Path to directory of test set")
    parser.add_argument('--model', type=str, default='./model/hydxj_model.h5',
                        help='Path to model.')
    parser.add_argument('--prediction_file', type=str, default='./result/test_results.csv',
                        help='Path to prediction file.')
    parser.add_argument('--nb_class', type=int, default=85,
                        help='number of people')
    parser.add_argument('--test_mode', type=int, default=0,
                        help='default 0 for the unknown dataset, 1 for known dataset')
    parser.add_argument('--test_dir', type=str, default='./af2019-ksyun-training-20190416',
                        help='when test_dataset is txt file, the path of image directory should be set')
    args = parser.parse_args()

    dataset = args.test_dataset
    if '.txt' not in dataset:
        base_dir = dataset + '/images/'
        dataset = read_csv(dataset)
    else:
        base_dir = args.test_dir

    # custom parameters
    nb_class = args.nb_class

    model = model()
    test(model, dataset, args.model, args.prediction_file, args.test_mode)

