import tensorflow as tf
from tensorflow.keras.layers import *
import librosa
import os
import numpy as np
import sklearn as skl
from sklearn import mixture
import matplotlib.pyplot as plt
import random
import sounddevice as sd
import scipy

def load_data(path):
    """"
    Load all the data in the folder path
    Input: path to the folder
    Output:
        - audios the audio in the folder
        - labels the ID of the speaker as an integer
        - sr sampling rate of the audio data
        - max_length the length of the largest audio in sample.
    """
    files = os.listdir(path)
    random.shuffle(files)
    #files = files[:100]
    audios = []
    labels = []
    max_length = 0
    print("loading audio")
    for file in files:
        audio, sr = librosa.load(path + file, sr=None)
        if audio.shape[0] > max_length:
            max_length = audio.shape[0]
        audios.append(audio)
        label = int(file.split('_')[0])
        labels.append(label)
    return audios, labels, sr, max_length

def audio_to_mfcc(audios, sr, max_length):
    """
    Compute the mfcc of the given audio
    Input:
        - audios: the audio
        - sr: ampling rate of the audio
        - max length: Length of the largest audio in sample
    Output: List of mfccs

    """
    print("computing mfcc")
    mfccs = []
    for audio in audios:
        pad_width = max_length - audio.shape[0]
        audio = np.pad(audio, (0, pad_width))
        mfcc = librosa.feature.mfcc(audio, sr=sr)
        mfccs.append(mfcc)
    return mfccs

def split_dataset(mfccs, labels):
    """
    Randomly split the dataset into three part:
        - a training set with 70% of the sample
        - a validation set with 20% of the sample
        - a test set with 10% of the sample
    Input:
        - mfccs: a ndarray of shape [sample, width, height]
        - labels: a ndarray of shape [sample, ]
    Output:
        mfccs_train, label_train, mfccs_val, label_val, mfccs_test, label_test.
        All ndarray of shape [sample, width, height] fo mfccs, [sample,] for labels.
    """
    nb_sample = labels.shape[0]
    idx = np.arange(nb_sample)
    np.random.shuffle(idx)
    print(idx)
    print("label shape" + str(labels.shape))
    print("mfccs shape" + str(mfccs.shape))
    mfccs = mfccs[idx]
    labels = labels[idx]
    train_sample = int(0.7 * nb_sample)
    val_sample = train_sample + int(0.2 * nb_sample)
    test_sample = val_sample + int(0.1 * nb_sample)
    return mfccs[:train_sample], labels[:train_sample], \
           mfccs[train_sample:val_sample], labels[train_sample:val_sample], \
           mfccs[val_sample:test_sample], labels[val_sample:test_sample]


def build_model(input_shape):
    """
    Build a Sequential CNN classifier.
    Input:
        - input shape of the data to be feed in. Must be a tuple (width, height, channel=1)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(10, (3, 3), 2, padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(20, (3, 3), 2, padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    return model

def SVM_build_model(features, labels):
    """
    Build and fit the SVM model on the provided data
    Input:
        features: ndarray of shape [sample, feature]
        label: ndarray of shape [sample, ]
    Output:
        - the trained model
    """
    SVM_model=skl.svm.SVC()
    print(features.shape)
    SVM_model.fit(features, labels)
    return SVM_model

def Gaussian_build_model(features, labels):
    """
    Build and fit the GMM model on the provided data
    Input:
        features: ndarray of shape [sample, feature]
        label: ndarray of shape [sample, ]
    Output:
        - the trained model
    """
    gaussian_model = skl.mixture.GaussianMixture(n_components=10)
    gaussian_model.fit(features, labels)
    return gaussian_model

if __name__ == '__main__':
    #Load the data
    data_path = 'recordings/'
    audios, labels, sr, max_length = load_data(data_path)
    labels = np.stack(labels)

    #One hot encode the label for the CNN
    labels = tf.keras.utils.to_categorical(labels)

    #Compute the mfccs
    mfccs = audio_to_mfcc(audios, sr, max_length)
    mfccs = np.stack(mfccs)
    mfccs = np.expand_dims(mfccs, axis=-1)

    print(mfccs.shape)

    #Build the CNN model
    input_shape = mfccs[0].shape
    model = build_model(input_shape)
    model.summary()
    tf.keras.utils.plot_model(model, 'cnn.png', show_shapes=True)
    model.compile('adam', tf.keras.losses.categorical_crossentropy, metrics=['categorical_accuracy'])

    #Split the dataset
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(mfccs, labels)

    #Train the CNN
    model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val),
              callbacks=[tf.keras.callbacks.TensorBoard(), tf.keras.callbacks.EarlyStopping(patience=10),
                        tf.keras.callbacks.ModelCheckpoint('model')])

    #Evaluate the model
    model.evaluate(x_test, y_test)
    cnn_pred = model.predict(x_test)
    cnn_pred = np.argmax(cnn_pred, axis=1)

    #Reformat data for the SVM and Gauss model

    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    x_val = np.reshape(x_val, (x_val.shape[0], -1))
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    y_val = np.argmax(y_val, axis=1)

    print("Otherwise we can try to use a faster method like SVM")

    #Build and train the SVM
    SVM_model = SVM_build_model(x_train, y_train)

    #Test the SVM
    y_SVM_predicted = SVM_model.predict(x_test)
    conf_matrix_SVM = skl.metrics.confusion_matrix(y_test, y_SVM_predicted)
    accuracy_SVM = skl.metrics.accuracy_score(y_test, y_SVM_predicted)

    #Build the GMM
    gauss_model = Gaussian_build_model(x_train, y_train)

    #Test the GMM
    y_pred_gauss = gauss_model.predict(x_test)
    conf_matrix_gauss = skl.metrics.confusion_matrix(y_test, y_pred_gauss)
    accuracy_gauss = skl.metrics.accuracy_score(y_test, y_pred_gauss)

    #Show result
    print("SVM")
    plt.figure()
    plt.matshow(conf_matrix_SVM)
    plt.title('SVM confusion matrix')
    plt.colorbar()
    plt.show()

    print(accuracy_SVM)
    print(conf_matrix_SVM)
    print("Gaussian model")
    print(accuracy_gauss)
    print(conf_matrix_gauss)
    plt.figure()
    plt.matshow(conf_matrix_gauss)
    plt.title('Gauss confusion matrix')
    plt.colorbar()
    plt.show()
    print('CNN')
    plt.figure()
    conf_matrix_CNN = skl.metrics.confusion_matrix(y_test, cnn_pred)
    plt.matshow(conf_matrix_CNN)
    plt.title('CNN confusion matrix')
    plt.colorbar()
    plt.show()
    print(conf_matrix_CNN)


    # For live demonstration. Does't work really well, as the dataset contains only 4 different speaker.
    # The CNN model is too specialized for these only 4 peoples.

    # fs = 8000 # Sample rate
    # seconds = 2  # Duration of recording

    # while(True):
    #     input("Press Enter to continue...")
    #     print("Say a number")
    #     myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    #     sd.wait()  # Wait until recording is finished
    #
    #     scipy.io.wavfile.write('test', fs, myrecording)
    #     myrecording, sr = librosa.load('test', sr=None)
    #     print(myrecording)
    #
    #     myrecording, _ = librosa.effects.trim(myrecording, top_db=30)
    #     myrecording = myrecording[:max_length]
    #     print("recording length" + str(myrecording.shape))
    #     print("max length " + str(max_length))
    #
    #     live_mfcc = audio_to_mfcc([myrecording], sr, max_length)
    #     live_mfcc = np.stack(live_mfcc)
    #     live_mfcc = np.expand_dims(live_mfcc, -1)
    #     print(live_mfcc.shape)
    #     cnn_pred = model.predict(live_mfcc)
    #     print("prediction: " + str(np.argmax(cnn_pred)))


