# Feature Extraction from specific layers in Tensorflow
# By Sagar Vakkala

#For now, I'll import the VGG19 model from the applications module of Keras
from keras.applications.vgg19 import VGG19

#Importing preprocessing
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

#Need to Import the Model method to define the VGG19 Instance as a Keras Model
from keras.models import Model
import numpy as np

#Making the base model Instance by Initiaiting the VGG19 
base_model = VGG19(weights='imagenet')

'''
Note: Models in Keras are saved in Json format. While, The weights are saved in .hdf5 format.
      A whole model can also be saved as .h5 model. But, If just the weights are loaded, It won't compile.
'''

model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_conv4').output)

img_path = 'example.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)

# Since the model expects a batch, Another dimension is added which consists the batch size which is then fed into preprocess_input()
x = np.expand_dims(x, axis=0)

# The end arrays have to be passed to preprocess_input() as different architectures will have different normalized scale values for the images.
# For ex: 0 to 1, -1 to 1 or just not normalized
x = preprocess_input(x)

block4_conv4_features = model.predict(x)
print(block4_conv4_features)