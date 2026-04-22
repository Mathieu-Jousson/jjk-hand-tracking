# 🤞 JJK Domain Expansion Detector (Jujutsu Kaisen)

---

## 🇫🇷 Français (English below)

### 📖 Description
Ce projet est un système de reconnaissance de gestes en temps réel basé sur l'univers de **Jujutsu Kaisen**. Déclenchement d'extensions du territoire à la reconnaissance des bons signes par vision par ordinateur et Machine Learning. 

### 🛠️ Caractéristiques Techniques
- **Vision ordinateur :** Google Mediapipe ("https://research.google/pubs/mediapipe-hands-on-device-real-time-hand-tracking/")
- **Modèle :** Random Forest, 182 features (88 points par main + 6 caractéristiques de liaison)
- **Audio :** Pygame

### 🚀 Installation et Lancement Rapide
Ouvrez votre terminal (Git Bash recommandé sur Windows) et exécutez ces commandes à la suite.


Bash :

```bash
# Cloner le projet
git clone https://github.com/Mathieu-Jousson/jjk-hand-tracking.git
cd jjk-hand-tracking

# Créer et activer un environnement virtuel
python -m venv venv
source venv/Scripts/activate 
# remplacer par ".\venv\Scripts\Activate.ps1" pour Windows Powershell
# remplacer par "source venv/bin/activate" pour Mac/Linux

# Installation des dépendances
pip install -r requirements.txt

# Lancement
python src/main.py
```




## 🇺🇸 English

### 📖 Description
This project is a real-time gesture recognition system based on the Jujutsu Kaisen universe. It triggers Domain Expansions by recognizing specific hand signs using Computer Vision and Machine Learning.

### 🛠️ Technical Specifications
- Computer Vision: Google Mediapipe ("https://research.google/pubs/mediapipe-hands-on-device-real-time-hand-tracking/")
- Model: Random Forest, 182 features (88 points per hand + 6 linking features)
- Audio: Pygame

### 🚀 Quick Start & Installation
Open your terminal (Git Bash recommended for Windows) and run the following commands.

Bash :

```Bash
# Clone the project
git clone https://github.com/Mathieu-Jousson/jjk-hand-tracking.git
cd jjk-hand-tracking

# Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate
# replace by ".\venv\Scripts\Activate.ps1" on Windows Powershell
# remplace by "source venv/bin/activate" on Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run
python src/main.py
```