from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load trained CNN model
model = load_model('models/skin_disease_model.h5')

# Define class names as per your model training
class_names = ['Acne and Rosacea Photos',
               'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
               'Atopic Dermatitis Photos',
               'Bullous Disease Photos',
               'Cellulitis Impetigo and other Bacterial Infections',
               'Eczema Photos',
               'Exanthems and Drug Eruptions',
               'Hair Loss Photos Alopecia and other Hair Diseases',
               'Herpes HPV and other STDs Photos',
               'Light Diseases and Disorders of Pigmentation',
               'Lupus and other Connective Tissue diseases',
               'Melanoma Skin Cancer Nevi and Moles',
               'Nail Fungus and other Nail Disease',
               'Poison Ivy Photos and other Contact Dermatitis',
               'Psoriasis pictures Lichen Planus and related diseases',
               'Scabies Lyme Disease and other Infestations and Bites',
               'Seborrheic Keratoses and other Benign Tumors',
               'Systemic Disease',
               'Tinea Ringworm Candidiasis and other Fungal Infections',
               'Urticaria Hives',
               'Vascular Tumors',
               'Vasculitis Photos',
               'Warts Molluscum and other Viral Infections']
# Function to predict the skin disease from the image
# Add this to the end of your predict_skin_disease.py file

def predict_skin_disease(image_path):
    try:
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        predicted_class = class_names[class_idx]

        return predicted_class
    except Exception as e:
        return f"Error: {str(e)}"



# Manually provide the image path here
image_path = "test/acne1.jpg"  # Change this to your image path

# Call the prediction function
predict_skin_disease(image_path)
