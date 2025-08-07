import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model

IMG_SIZE = 224 

# Paths
DATA_DIR = "data/images"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Get class names (disease types) from directory names
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

def predict_disease_by_saved_model(image_path, model):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    return {
        'disease': class_names[predicted_class],
        'confidence': float(confidence),
        'probabilities': {class_names[i]: float(p) for i, p in enumerate(prediction[0])}
    }

# Example usage of prediction function
# Replace with actual image path for testing
test_image_path = "D:\Projects\mango-leaf-classifer\data\external_test_images\sample_2.jpg"


model_path = os.path.join(MODEL_DIR, f'model_mobilenetv2_20250806_015706.h5')
model = load_model(model_path)
print(f"Loaded model from {model_path}")
if os.path.exists(test_image_path):
    result = predict_disease_by_saved_model(test_image_path, model)
    print("\nPrediction Result:")
    print(f"Predicted Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nAll probabilities:")
    for disease, prob in result['probabilities'].items():
        print(f"{disease}: {prob:.2%}")