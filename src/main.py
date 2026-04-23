from unittest import result
import numpy as np
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import joblib
import pygame
import os

from building_dataset import FeatureExtractor


# --- CONFIGURATION & CHARGEMENT ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ML_PATH = os.path.join(ROOT_DIR, "models", "jjk_model_20260422-1043.joblib")
MODEL_TASK_PATH = os.path.join(ROOT_DIR, "models", "hand_landmarker.task")

jjk_model = joblib.load(MODEL_ML_PATH)
class_names = ["Idle", "Gojo", "Sukuna", "Mahito", "Megumi"]

# Connexions pour le dessin du squelette
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# --- INITIALISATION AUDIO ---
pygame.mixer.init()
# Dictionnaire reliant l'ID du signe au fichier son
sounds = {
    1: pygame.mixer.Sound(os.path.join(ROOT_DIR, "sounds", "gojo.wav")),
    2: pygame.mixer.Sound(os.path.join(ROOT_DIR, "sounds", "sukuna.wav")),
    3: pygame.mixer.Sound(os.path.join(ROOT_DIR, "sounds", "mahito.wav")),
    4: pygame.mixer.Sound(os.path.join(ROOT_DIR, "sounds", "megumi.wav"))
}


# --- BOUCLE PRINCIPALE ---
def main():
    extractor = FeatureExtractor()
    base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.8,
        min_hand_presence_confidence=0.8,
        min_tracking_confidence=0.8
    )
    detector = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)


    last_prediction=0
    consecutive_frames=0


    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        feat_L, feat_R = [-10.0]*88, [-10.0]*88
        lm_L, lm_R = None, None
        sc_L, sc_R = 0, 0
        all_pts = []


        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                side = result.handedness[i][0].category_name 
                
                f, s = extractor.get_hand_features(landmarks)
                
                color_skel = (255, 255, 255)

                if side == 'Left':
                    feat_L, lm_L, sc_L = f, landmarks, s
                else:
                    feat_R, lm_R, sc_R = f, landmarks, s

                # Dessin du squelette
                for conn in HAND_CONNECTIONS:
                    p1, p2 = landmarks[conn[0]], landmarks[conn[1]]
                    cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), color_skel, 2)
                for lm in landmarks:
                    all_pts.append((int(lm.x*w), int(lm.y*h)))

            # Calcul des 6 features Inter-mains
            feat_inter = extractor.get_inter_hand_features(lm_L, lm_R, sc_L, sc_R)
            
            # Assemblage du vecteur final (182)
            full_vector = feat_L + feat_R + feat_inter

        # C. Prédiction et Box
            input_data = np.array(full_vector).reshape(1, -1)
            probabilities = jjk_model.predict_proba(input_data)[0]
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]

            
            # Si la prédiction est identique à la précédente et n'est pas "Repos"
            if prediction == last_prediction and prediction != 0:
                consecutive_frames += 1
            else:
                consecutive_frames = 0
                last_prediction = prediction

            # Déclenchement au bout de 5 frames consécutives
            if consecutive_frames >= 5:
                # Vérifier si un son est déjà en train de jouer
                if not pygame.mixer.get_busy():
                    if prediction in sounds.keys():
                        sounds[prediction].play()
                        consecutive_frames = 0

            # On vérifie qu'on a bien des points avant de calculer la box
            if confidence > 0.80 and prediction != 0 and len(all_pts) > 0:
                try:
                    # Conversion en array numpy pour les calculs
                    pts_array = np.array(all_pts)
                    
                    # On trouve les coins (axis=0 cherche le min/max global pour X et Y)
                    x_min, y_min = np.min(pts_array, axis=0)
                    x_max, y_max = np.max(pts_array, axis=0)

                    # Ajout d'une marge de 20 pixels
                    x_min, y_min = x_min - 20, y_min - 20
                    x_max, y_max = x_max + 20, y_max + 20
                    
                    # Choix de la couleur selon le territoire
                    colors = [(0,255,0), (255,0,0), (0,0,255), (0,255,255), (255,0,255)]
                    box_color = colors[prediction % len(colors)]

                    # Dessin de la boîte et du texte
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), box_color, 2)
                    
                    label = f"{class_names[prediction]} {confidence*100:.0f}%"
                    cv2.putText(frame, label, (int(x_min), int(y_min)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                except Exception as e:
                    print(f"Erreur visuelle : {e}")

        cv2.imshow("Ryooiki tenkai", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
        if cv2.getWindowProperty("Ryooiki tenkai", cv2.WND_PROP_VISIBLE) < 1: break

    detector.close(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()