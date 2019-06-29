from keras.models import load_model
from Task1 import ImgReader
import numpy as np
import pickle

model = load_model("./Lunar_Lander_models/Task1.hdf5")
with open("./Lunar_Lander_models/label_encoder.pkl", 'rb') as fh:
    label_encoder = pickle.load(fh)
img = ImgReader("/home/hugh/Adv_ML/Assignment2/frames/LunarLanderFramesPart1/LunarLanderFramesPart1", img_width=100, img_height=100)

img.process_image_files(0, 600, False)
#print(np.bincount(img.y_data_part))
X, y = img.get_random_sample()
X = X.reshape(len(X), 100, 100, 1)
prediction = model.predict(X)
accuracy = 0

for i, predict in enumerate(prediction):

    if np.argmax(predict) == y[i]:
        accuracy += 1

print("Average classification accuracy:", accuracy/len(prediction))
