import tensorflow as tf
from keras import Model

def predict_mask(model : Model, file_path : str) :
    
    img = tf.io.read_file(file_path)
    tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32) 
    input_tensor = tf.expand_dims(tensor, axis=0)
    
    return tensor, model.predict(input_tensor)