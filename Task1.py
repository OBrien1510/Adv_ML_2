import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM
from keras import backend as K
import sklearn
from sklearn.model_selection import train_test_split
import re
import pickle
import os
import cv2
import random
import numpy as np

# The file to train the supervised model. There is also a test when it finished training
# model_test.py also implements a test on a pretrained model

class ImgReader:

    def __init__(self, filepath, img_width=150, img_height=150, sample_size=0.01):

        m = re.compile("frame_(.+).jpeg")
        self.filepath = filepath
        self.files = sorted([i for i in os.listdir(filepath)])
        self.img_width = img_width
        self.processed = False
        self.img_height = img_height
        self.y_data = [m.match(file) for i, file in enumerate(sorted(os.listdir(filepath)))]
        self.y_data = [int(i.group(1)[-1]) for i in self.y_data]
        self.balance_data()
        self.sample_size = sample_size
        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, img_width, img_height)
        else:
            self.input_shape = (img_width, img_height, 3)
        self.model = self.get_model()
        self.X_data = np.ndarray((0, self.input_shape[0], self.input_shape[1]), dtype=np.float)
        self.y_data = np.array(self.y_data)
        # perform random sampling
        count = int(len(self.files) * self.sample_size)
        random_index = np.random.choice(np.arange(len(self.files)), count, replace=False)
        self.files = np.array(self.files)[random_index]
        self.y_data = np.array(self.y_data)[random_index]

    def balance_data(self):

        min_class = np.min(np.bincount(self.y_data))
        idx = np.hstack([np.random.choice(np.where(self.y_data == l)[0], int(min_class), replace=True) for l in np.unique(self.y_data)])
        self.y_data = np.array(self.y_data)[idx]
        self.files = np.array(self.files)[idx]


    def get_batch_index(self, batch_size):

        count = int(len(self.files))
        index = list(range(batch_size, count, batch_size))
        index.append(batch_size*len(index) + (count - batch_size*(len(index))))
        return index

    def process_image_files(self, batch_start, batch_end, all):

        self.X_data_part = list()
        self.y_data_part = list()

        if all == True:
            batch_start = 0
            batch_end = len(self.files)

        for i, image in enumerate(self.files[batch_start:batch_end]):

            current_filepath = self.filepath+"/"+image
            image = cv2.imread(current_filepath, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (self.img_height, self.img_width), interpolation=cv2.INTER_CUBIC)

            # load image in grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if K.image_data_format() == 'channels_first':
                image = np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)

            # Add image data to data array and normalise
            self.X_data_part.append(image/255)
            self.y_data_part.append(self.y_data[batch_start+i])

            if i % 100 == 0:
                print("Processed %d images" % i)


        self.X_data_part = np.array(self.X_data_part)
        self.y_data_part = np.array(self.y_data_part)
        self.processed = True
        print("X Shape:", self.X_data_part.shape)
        print("y shape:", self.y_data_part.shape)


    def view_sample(self):

        if self.processed:
            # view random photo
            i_rand = random.randint(0, self.X_data_part.shape[0])
            cv2.imshow(self.y_data[i_rand], self.X_data_part[i_rand])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print("Process images before trying to view")


    def get_model(self):

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(self.img_height, self.img_width, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256))
        #model.add(LSTM(64, input_shape=(1, 64), activation="relu"))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(np.unique(self.y_data))))
        model.add(Activation('softmax'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


        return model


    def get_random_sample(self, sample_size=1):


        random_index = np.random.choice(np.arange(len(self.X_data_part)), int(len(self.X_data_part)*sample_size), replace=False)
        return np.array(self.X_data_part)[random_index], np.array(self.y_data)[random_index]



    def train(self):


        self.X_data_part = self.X_data_part.reshape(len(self.X_data_part), self.img_height, self.img_width, 1)
        # Convert class vectors to binary class matrices.

        X_train, X_test, y_train, y_test = train_test_split(self.X_data_part, self.y_data_part, random_state=0, test_size=0.1,
                                                            train_size=0.9)
        self.y_train_encoder = sklearn.preprocessing.LabelEncoder()


        y_train_num = self.y_train_encoder.fit_transform(y_train)
        classes_num_label = dict()

        for idx, lbl in enumerate(self.y_train_encoder.classes_):
            classes_num_label[idx] = lbl

        # save the label encoding table so we can convert preditctitions back to numbered actions later
        with open("./Lunar_Lander_models/label_encoder.pkl", 'wb+') as fh:
            pickle.dump(classes_num_label, fh)

        y_train_wide = keras.utils.to_categorical(y_train_num, len(np.unique(self.y_data_part)))
        self.history = self.model.fit(X_train, y_train_wide,
                                        batch_size=64,
                                        epochs=50,
                                        verbose=1,
                                        validation_split=0.2,
                                        shuffle=True,)

        y_test_encoder = sklearn.preprocessing.LabelEncoder()
        y_test_num = y_test_encoder.fit_transform(y_test)
        y_test_wide = keras.utils.to_categorical(y_test_num, len(np.unique(self.y_data_part)))

        return X_test, y_test_wide


if __name__ == "__main__":

    filepath = "/home/hugh/Adv_ML/Assignment2/frames/LunarLanderFramesPart1/LunarLanderFramesPart1"
    # full dataset was talking very long and after a while the accuracy just leveled out so use a smaller subset
    img = ImgReader(filepath=filepath, sample_size=0.2, img_height=100, img_width=100)
    batch_size = img.get_batch_index(600)
    batch_1 = 0
    X_test = []
    y_test = []
    print(batch_size)

    # load images into memory in batches and iteratively train model on each batch
    for i, batch in enumerate(batch_size):

        batch_2 = batch
        if batch_1 == batch_2:
            break

        img.process_image_files(batch_1, batch_2, False)

        # train method also returns the test sets for accuracy tests
        batch_X_test, batch_y_test = img.train()
        X_test.append(batch_X_test)
        y_test.append(batch_y_test)
        batch_1 = batch_2


    av_accuracy = []
    accuracy = 0
    total = 0
    for i, array in enumerate(X_test):
        predictions = img.model.predict(np.array(array))
        #print(predictions.shape)
        for j, prediction in enumerate(predictions):
            max_index_x = np.argmax(prediction)
            max_index_y = np.argmax(y_test[i][j])
            if max_index_x == max_index_y: accuracy += 1
            total += 1

    print("Average accuray on test set:", accuracy/total)

    img.model.save("./Lunar_Lander_models/Task1.hdf5")
