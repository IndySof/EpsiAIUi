
# Espia

**Espia** est un projet de classification d'images OCT (Tomographie par Cohérence Optique) utilisant un modèle de réseau de neurones convolutif (CNN) avec TensorFlow et Keras. Le modèle, nommé `EspiAI`, est conçu pour identifier diverses pathologies de la rétine à partir d'images OCT et utilise le fichier de modèle `best.hdf5`.

## Pathologies Détectées

EspiAI est capable de détecter les pathologies suivantes :

- **Dégénérescence Maculaire Liée à l'Âge (DMLA)** : caractérisée par une membrane néovasculaire et un fluide sous-rétinien.
- **Œdème Maculaire Diabétique (OMD)** : épaississement de la rétine avec du fluide intrarétinien.
- **Néovaisseaux Choroïdiens (NVC)** : associée à des drusen dans les stades précoces de la DMLA.
- **Rétine Normale** : structure rétinienne intacte sans signes de fluides ou d'œdème.

<p align="center">
  <img src="https://github.com/IndySof/EpsiAI/blob/master/Dataset/test/DMLA/DRUSEN-8549730-2.jpeg" alt="Exemple DMLA" width="200"/>
  <img src="https://github.com/IndySof/EpsiAI/blob/master/Dataset/train/OMD/DME-1072015-2.jpeg" alt="Exemple OMD" width="200"/>
</p>

## Technologies Utilisées

- **Streamlit** : pour créer une application web interactive permettant aux utilisateurs de charger des images OCT et d'obtenir une prédiction.
- **TensorFlow et Keras** : pour construire et entraîner le modèle de classification d'images.

## Démarrage de l'Application

1. **Installation des dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Lancement de l'application** :
   ```bash
   streamlit run app.py
   ```

   Cette commande démarre l'application Streamlit, où vous pouvez charger une image OCT pour obtenir une prédiction de la pathologie.

## Structure des Fichiers

- `app.py` : script principal pour l'application Streamlit.
- `best.hdf5` : fichier de modèle entraîné pour la classification des images OCT.
- `Dataset/` : contient les images d'entraînement et de test utilisées pour entraîner le modèle.

## Utilisation

1. **Charger une image OCT** dans l'application Streamlit.
2. **Recevoir une prédiction** : EspiAI identifiera la pathologie parmi les catégories définies.

## Auteur

EspiAI a été développé pour faciliter le diagnostic des pathologies rétiniennes en utilisant des techniques de deep learning appliquées aux images OCT.
