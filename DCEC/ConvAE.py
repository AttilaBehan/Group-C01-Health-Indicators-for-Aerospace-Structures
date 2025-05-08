from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np


# def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
#     model = Sequential()
#     if input_shape[0] % 8 == 0:
#         pad3 = 'same'
#     else:
#         pad3 = 'valid'
#     model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))
#
#     model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))
#
#     model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))
#
#     model.add(Flatten())
#     model.add(Dense(units=filters[3], name='embedding'))
#     model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))
#
#     model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
#     model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))
#
#     model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))
#
#     model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
#     model.summary()
#     return model

def dense_CAE(input_shape=(6,), hidden_units=[64, 32, 16]):
    input_layer = Input(shape=input_shape, name='input')
    x = Dense(hidden_units[0], activation='relu')(input_layer)
    x = Dense(hidden_units[1], activation='relu')(x)
    encoded = Dense(hidden_units[2], activation='relu', name='embedding')(x)

    x = Dense(hidden_units[1], activation='relu')(encoded)
    x = Dense(hidden_units[0], activation='relu')(x)
    decoded = Dense(input_shape[0], name='decoder')(x)

    model = Model(inputs=input_layer, outputs=decoded)
    return model

if __name__ == "__main__":
    from time import time

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='aedata', choices=['aedata', 'mnist', 'usps'])
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_dir', default='results/temp', type=str)
    args = parser.parse_args()
    print(args)

    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from datasets import load_mnist, load_usps, load_aedata
    if args.dataset == 'aedata':
        x, y = load_aedata()
    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'usps':
        x, y = load_usps('data/usps')

    # define the model
    model = dense_CAE(input_shape=x.shape[1:], hidden_units=[64, 32, 16])
    plot_model(model, to_file=args.save_dir + '/%s-pretrain-model.png' % args.dataset, show_shapes=True)
    model.summary()

    # compile the model and callbacks
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='mse')
    from tensorflow.keras.callbacks import CSVLogger
    csv_logger = CSVLogger(args.save_dir + '/%s-pretrain-log.csv' % args.dataset)

    # begin training
    t0 = time()
    model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
    print('Training time: ', time() - t0)
    model.save(args.save_dir + '/%s-pretrain-model-%d.h5' % (args.dataset, args.epochs))

    # extract features
    feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
    features = feature_model.predict(x)
    print('feature shape=', features.shape)

    # use features for clustering
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.n_clusters)

    features = np.reshape(features, newshape=(features.shape[0], -1))
    pred = km.fit_predict(features)
    from . import metrics
    print('acc=', metrics.acc(y, pred), 'nmi=', metrics.nmi(y, pred), 'ari=', metrics.ari(y, pred))
