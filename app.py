import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configurer le titre et l'icône de l'onglet du navigateur
st.set_page_config(page_title="EpsiAI - Classification d'OCT", page_icon=":microscope:")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./best.hdf5')
    return model

def predict_class(image, model):
    image = tf.image.resize(image, [32, 32])
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

model = load_model()
st.title('EpsiAI - Classification d\'OCT')

file = st.file_uploader("Téléchargez une image d'OCT", type=["jpg", "png"])

if file is None:
    st.text('En attente du téléchargement....')
else:
    slot = st.empty()
    slot.text('Analyse en cours....')

    test_image = Image.open(file)
    if test_image.mode != "RGB":
        test_image = test_image.convert("RGB")
    st.image(test_image, caption="Image d'entrée", width=400)

    test_image_np = np.asarray(test_image) / 255.0
    pred = predict_class(test_image_np, model)
    class_names = ['Dégénérescence Maculaire Liée à l\'Âge', 'Rétine Normale', 'Néovaisseaux Choroïdiens', 'Œdème Maculaire Diabétique']
    result = class_names[np.argmax(pred)]
    output = 'La pathologie détectée est : ' + result
    slot.text('Terminé')
    st.success(output)
