import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image
import json
import glob 
from tensorflow.keras import layers

import argparse





parser = argparse.ArgumentParser(description='Image Classifier - Prediction Part')
parser.add_argument('--input', default='./test_images/hard-leaved_pocket_orchid.jpg', action="store", type = str, help='image path')
parser.add_argument('--model', default='1650795886.h5', action="store", type = str, help='Model.h5 path')
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help='Number of most likely classes')
parser.add_argument('--category_names', dest="category_names", action="store", default='label_map.json', help='categories  names json file')


arg_parser = parser.parse_args()

image_path = arg_parser.input
model_path = arg_parser.model
topk = arg_parser.top_k
category_names = arg_parser.category_names

batch_size = 32
image_size = 224



def process_image(image): 
   
    image = tf.cast(image, tf.float32)
    image= tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    
    return image
    

def predict(image_path, model, top_k=5):
    
    image = Image.open(image_path)
    image = np.asarray(image)
    image = np.expand_dims(image,  axis=0)
    image = process_image(image)
    prob_list = model.predict(image)

    
    classes = []
    probs = []
    
    rank = prob_list[0].argsort()[::-1]
    
    for i in range(top_k):
        
        index = rank[i] + 1
        cls = class_names[str(index)]
        
        probs.append(prob_list[0][index])
        classes.append(cls)
    
    return probs, classes


if __name__ == '__main__':
    print('Pridection')
    
    model = tf.keras.models.load_model(model_path ,custom_objects={'KerasLayer':hub.KerasLayer} )
    with open(category_names, 'r') as f:
        class_names = json.load(f)
   
    probs, classes = predict(image_path, model, topk)

    print(probs)
    print(classes)

