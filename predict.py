from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import pickle
import cv2


# load model
imodel = "./model/modelku.h5"

# load label
ilabel = "./output/label.lb"

# image
iimage = "./test/blackspot.jpg"

# load model yang sudah terlatih
print("[INFO] loading model and label binarizer...")
model = load_model(imodel)
lb = pickle.loads(open(ilabel, "rb").read())

# load gambar jadikan RGB
frame = cv2.imread(iimage)
output = frame.copy()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (224, 224)).astype("float32")

#image = image.astype("float") / 255.0
frame = img_to_array(frame)
frame = np.expand_dims(frame, axis=0)

#prediksi
proba = model.predict(frame)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]
cv2.putText(output, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (0, 255, 0), 1)

#tampilkan
cv2.imshow("Output", output)
cv2.waitKey(0)
