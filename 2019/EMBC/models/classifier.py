
import matplotlib.pyplot as plt
import os
from scipy import stats, random
from keras.models import Sequential, load_model
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from nnom_utils import *
from pre_process import *

def windows(data, size):
    start = 0
    while start < data.count():
        yield np.int(start), np.int(start + size)
        start += (size / 4)  # windows overlay. 1/2 or 1/4 is almost the same (4 is better)

def data_segment(data, window_size=40, selected_subjects=None, raw_sample_rate = 100, re_sample_rate = 20, data_type='scale'):

    de_sampling = np.int(raw_sample_rate / re_sample_rate)
    labels = np.empty((0))
    lenght = len(data['timestamp'])
    print('original data lenght:', lenght)

    # de sampling
    data = data[data.index % de_sampling == 0].reset_index()
    lenght = len(data['timestamp'])
    print('de-sampling data lenght:', lenght)

    # select data
    selected = []
    data_channel = 0
    if 'scale' in data_type:
        # everythings is normal! go get the data
        selected.append('quantized_acc')
        selected.append('quantized_rotate')
        selected.append('quantized_force')
        data_channel+=3

    if 'acc' in data_type:
        selected.append('quantized_acc_x')
        selected.append('quantized_acc_y')
        selected.append('quantized_acc_z')
        data_channel += 3

    if 'gyro' in data_type:
        selected.append('quantized_gyro_x')
        selected.append('quantized_gyro_y')
        selected.append('quantized_gyro_z')
        data_channel += 3
    # place holders
    segments = np.empty((0, window_size, data_channel)).astype(np.float16) # should be enough, the original is 8bit..integral

    # convert the whole select dataframe to nparray to improve efficiency
    np_data = data[selected][:].values.astype(np.float16)
    np_data = np_data[np.newaxis, :, :]

    cross_drop = 0
    count = 0
    for (start, end) in windows(data['timestamp'], window_size):
        if (count % (lenght // (10 * window_size)) == 0):
            print('segment index:', start, '  completed', int(start / lenght * 100 + 1), '%')
        count += 1

        # CHECK and save data data
        if (len(data['timestamp'][start:end]) != window_size):
            continue
        # do not save the data with crossing exercise
        if (data["T"][start] != np.average(data["T"][start:end])):
            cross_drop += 1
            continue
        # do not save the data with crossing different device
        if (data["sensor_addr"][start:end].unique().size > 1 ):
            cross_drop += 1
            continue
        # check the current subject is the selected
        if(selected_subjects is not None):
            S = data["S"][start]
            if str(S) not in selected_subjects:
                continue

        # do the job
        segments = np.vstack([segments, np_data[:, start:end]])
        labels = np.append(labels, stats.mode(data["T"][start:end])[0][0])

    print('segmented pices:', labels.size)
    print(labels)
    print("drop frames due to crossing:", cross_drop)
    return segments, labels



def train(x_train, y_train, x_test, y_test, batch_size= 128, epochs = 100, model_path="models/best.h5"):
    # shuffle
    permutation = np.random.permutation(y_train.shape[0])
    x_train = x_train[permutation, :, :]
    y_train = y_train[permutation]

    print("x_train shape", x_train.shape)
    print("y_train shape", y_train.shape)

    # to record all the shift
    m = nnom()

    inputs = Input(shape=x_train.shape[1:])
    x = Conv1D(8, kernel_size=(11), strides=(2), padding='same')(inputs)
    x = m.fake_clip(x, frac_bit=0, bit=8)(x)
    x = ReLU()(x)
    x = MaxPool1D(2, strides=2)(x)
    x = Dropout(0.2)(x)

    # inception - 1
    x1 = Conv1D(8, kernel_size=(7), strides=(1), padding="same")(x)
    x1 = m.fake_clip(x1, frac_bit=0)(x1)
    x1 = ReLU()(x1)
    x1 = MaxPool1D(2, strides=2)(x1)
    x1 = Dropout(0.2)(x1)

    # inception - 2
    x2 = Conv1D(8, kernel_size=(3), strides=(1), padding="same")(x)
    x2 = m.fake_clip(x2, frac_bit=0)(x2)
    x2 = ReLU()(x2)
    x2 = MaxPool1D(2, strides=2)(x2)
    x2 = Dropout(0.2)(x2)

    # inception - 3
    x3 = Conv1D(8, kernel_size=1, strides=1)(x)
    x3 = MaxPool1D(2, strides=2)(x3)
    x3 = Dropout(0.2)(x3)

    # concate all inception layers
    x = concatenate([x1, x2], axis=-1)  # This 2 lines are same as x = concatenate([x1, x2, x3], axis=-1)
    x = concatenate([x, x3], axis=-1)

    # conclusion
    x = Conv1D(24, kernel_size=(3), strides=(1), padding="same")(x)
    x = m.fake_clip(x, frac_bit=0)(x)
    x = ReLU()(x)
    x = MaxPool1D(2, strides=2)(x)
    x = Dropout(0.5)(x)

    # our netowrk is not that deep, so a hidden fully connected layer is introduce
    x = Flatten()(x)
    x = Dense(64)(x)
    x = m.fake_clip(x, frac_bit=0)(x)
    x = ReLU()(x)
    x = Dropout(0.6)(x)
    x = Dense(4)(x)
    x = m.fake_clip(x, frac_bit=0)(x)
    predictions = Softmax()(x)
    #predictions = fake_clip_min_max(max=1, min=-1)(predictions) # this might improve accuracy on MCU side

    model = Model(inputs=inputs, outputs=predictions)

    model.compile( loss='categorical_crossentropy',
                  optimizer='Adam',
                  #optimizer='Adadelta',
                  #optimizer='RMSprop',
                  #optimizer='SGD',
                  metrics=['accuracy'])

    model.summary()

    # save best
    checkpoint = ModelCheckpoint(filepath=model_path,
            monitor='val_acc',
            verbose=0,
            save_best_only='True',
            mode='auto',
            period=1)
    callback_lists = [checkpoint]

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                        verbose=2, validation_data=(x_test, y_test), callbacks=callback_lists)
    # save output shifts
    m.save_shift()
    # free the session to avoid nesting naming while we load the best model after.
    del model
    K.clear_session()
    return history

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    subjects_train = [ '2', '3', '4', '6', '7', '8', '9', '10', '11', '12']
    subjects_test = ['1', '5', '13']

    epochs = 100
    segment_len = 128
    sampling_frequncy = 50

    test_names = "test"
    model_name = "models/" + test_names +'.h5'
    evl_name = 'evl' + test_names + '.txt'
    data_file_name = 'merged_all.csv'

    # get data
    pdata = smartband_preprocess()

    # segment or load data
    if (os.path.exists("train_data.npy") and os.path.exists("test_data.npy")):
        train_data = np.load("train_data.npy")
        train_labels = np.load("train_labels.npy")
        test_data = np.load("test_data.npy")
        test_labels = np.load("test_labels.npy")
        print("data from pre-segmented file")

    else:
        # load raw data and generate pre-process data
        if (os.path.exists(data_file_name)):
            dataset = pd.read_csv(data_file_name)
            print("data read from existing file")
        else:
            # cut first 5 seconds
            pdata.generate_merged_file('resampled', output_file_path=data_file_name, method='mid', cut=100 * 2,
                                       normalize=True, sampling_rate=100, lowpass_cutoff=25)
            dataset = pd.read_csv(data_file_name, header=0)

        train_data, train_labels = data_segment(dataset, segment_len, selected_subjects=subjects_train,
                                        raw_sample_rate=100, re_sample_rate=sampling_frequncy, data_type='scale')
        train_labels = np.asarray(pd.get_dummies(train_labels), dtype=np.int8)

        test_data, test_labels = data_segment(dataset, segment_len, selected_subjects=subjects_test,
                                        raw_sample_rate=100, re_sample_rate=sampling_frequncy, data_type='scale')
        test_labels = np.asarray(pd.get_dummies(test_labels), dtype=np.int8)

        np.save("train_data.npy", train_data)
        np.save("train_labels.npy", train_labels)
        np.save("test_data.npy", test_data)
        np.save("test_labels.npy", test_labels)
        print("save segmented data to file")

    x_train = train_data
    y_train = train_labels
    x_test = test_data
    y_test = test_labels

    # test, ranges
    print("train range", np.max(x_train[:, :, 0:3]), np.min(x_train[:, :, 0:3]))
    print("test range", np.max(x_test[:, :, 0:3]), np.min(x_test[:, :, 0:3]))

    # generate binary test data
    generate_test_bin(x_test, y_test, name='couch_potatos_test_data.bin')
    generate_test_bin(x_train, y_train, name='couch_potatos_train_data.bin')

    # train model
    history = train(x_train,y_train, x_test, y_test, batch_size=128, epochs=epochs, model_path=model_name)

    # get best model
    model = load_model(model_name)

    # save weight
    generate_weights(model)

    # test, show the output ranges
    layers_output_ranges(model, x_train)

    # evaluate
    evaluate_model(model, x_test, y_test, to_file=evl_name, running_time=True)


	acc = history.history['acc']
	val_acc = history.history['val_acc']

	plt.plot(range(0, epochs), acc, color='red', label='Training acc')
	plt.plot(range(0, epochs), val_acc, color='green', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

