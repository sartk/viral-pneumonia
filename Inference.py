import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
import PIL
from PIL import Image
from skimage.transform import resize

from tensorflow.keras.mixed_precision import experimental as mixed_precision
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.config.optimizer.set_experimental_options(
    {"auto_mixed_precision": True})

path = ''
model = load_model(path + 'COVID_val80.h5')
def infer(image):
    y = model.predict(image.reshape(1,250,250,3))[0]
    print(y)
    pred = np.argmax(y)

    if pred == 0:
        n = 'Normal'
    elif pred ==1:
        n = 'Bacterial'
    elif pred == 2:
        n = 'Viral'

    #returns diagnosis and confidence
    return (n, y[pred])

#will take any image(as np array) in any shape(3d)(width,height,channel(s))
def image_process(i):
    image = i
    base = 250
    array_dimension = [1,base,base,1]
    full_image_array = np.zeros(array_dimension)

    width = image.shape[0]
    height = image.shape[1]
    max_value = [width,height]
    max_value = max(max_value)
    ratio = base/max_value
    width_d = int(width*ratio)
    height_d = int(height*ratio)
    int_max = max(width_d,height_d)
    if int_max < base:
        if int_max == width_d:
            width_d = base
        elif int_max == height_d:
            height_d = base
        elif int_max == width_d == height_d:
            width_d = base
            height_d = base
    print(width,height,width_d, height_d)
    image = np.array(resize(image,(width_d,height_d)))
    zeros = np.zeros((base,base))
    zeros[:image.shape[0], :image.shape[1]] = (image if len(image.shape)==2 else image[:,:,0])
    image = zeros

    '''if(len(image.shape)==3):
        full_image_array = np.concatenate([full_image_array,image[:,:,0].reshape(array_dimension)],axis=0)
    else:'''
    full_image_array = np.concatenate([full_image_array,image.reshape(array_dimension)],axis=0)

    X = full_image_array[1:]
    X = X.reshape((X.shape[0],250,250,1)) * np.ones((1,1,1,3), dtype='uint8')
    return X
