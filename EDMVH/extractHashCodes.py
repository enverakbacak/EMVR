#coding=utf-8
from keras.models import Sequential,Input,Model,InputLayer
from keras.models import model_from_json
from keras.models import load_model
import numpy as np    # for mathematical operations

np.set_printoptions(linewidth=1024)


json_file = open('./models/64_3_model.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights("./models/64_3_weights.h5")
model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)

X = []
X = np.load(open("NP3.npy"))
X.shape
print(X.shape[:])
X = X.reshape(X.shape[0], X.shape[1] , X.shape[2] * X.shape[3] * X.shape[4])

features = model.predict(X, batch_size=64, verbose=0, steps=None) #     TRY THIS ALSO XX = model.predict(X, batch_size)
features = features.astype(float)
np.savetxt('hashCodes/features_64_3.txt',features, fmt='%f')

features = features > 0.5
features = features.astype(int)
np.savetxt('hashCodes/hashCodes_64_3.txt',features, fmt='%d')



