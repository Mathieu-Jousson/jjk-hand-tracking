import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os



# --- CONFIGURATION DES CHEMINS ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'hand_landmarker.task')
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")
DATASET_PATH = os.path.join(DATASET_DIR, "idle.csv")

NUM_FEATURES_PER_HAND = 88
NUM_INTER_FEAT = 6
TOTAL_FEAT = (NUM_FEATURES_PER_HAND * 2) + NUM_INTER_FEAT # 182

# --- CLASSE D'EXTRACTION DE CARACTÉRISTIQUES ---
class FeatureExtractor:
    def __init__(self):
        # Topologie des doigts pour les angles (3 angles par doigt)
        self.finger_indices = [
            [0, 1, 2, 3, 4],    # Pouce
            [0, 5, 6, 7, 8],    # Index
            [0, 9, 10, 11, 12], # Majeur
            [0, 13, 14, 15, 16],# Annulaire
            [0, 17, 18, 19, 20] # Auriculaire
        ]
        # Points des extrémités pour les distances
        self.tips = [4, 8, 12, 16, 20]

    def _get_angle(self, a, b, c):
        """Calcule l'angle entre les segments [ab] et [bc]"""
        v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def get_hand_features(self, landmarks):
        """Extrait les 88 features pour une main donnée"""
        # 1. Nettoyage : Centrage sur le poignet (point 0)
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        # Échelle : Distance poignet -> base du majeur (point 9)
        ref_p = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
        scale = np.linalg.norm(ref_p - wrist) + 1e-6
        
        # 2. Coordonnées (63 features)
        coords = []
        for lm in landmarks:
            coords.extend([
                (lm.x - wrist[0]) / scale,
                (lm.y - wrist[1]) / scale,
                (lm.z - wrist[2]) / scale
            ])
            
        # 3. Angles de flexion (15 features)
        angles = []
        for finger in self.finger_indices:
            for i in range(len(finger) - 2):
                angles.append(self._get_angle(landmarks[finger[i]], landmarks[finger[i+1]], landmarks[finger[i+2]]))
                
        # 4. Distances inter-extrémités (10 features)
        dists = []
        for i in range(len(self.tips)):
            for j in range(i + 1, len(self.tips)):
                p1 = np.array([landmarks[self.tips[i]].x, landmarks[self.tips[i]].y, landmarks[self.tips[i]].z])
                p2 = np.array([landmarks[self.tips[j]].x, landmarks[self.tips[j]].y, landmarks[self.tips[j]].z])
                dists.append(np.linalg.norm(p1 - p2) / scale)
                
        return (coords + angles + dists), scale


    # distances et angle inter-mains (6 features)
    def get_inter_hand_features(self, landmarks_L, landmarks_R, scale_L, scale_R):

        # Si une des mains est absente 
        if landmarks_L is None or landmarks_R is None:
            return [-10.0] * 6 
        
        # 1 : distances intermains 
        inter_distances=[]
        avg_scale=0.5*(scale_L + scale_R)
        for i in self.tips:
            pl=np.array([landmarks_L[i].x, landmarks_L[i].y, landmarks_L[i].z])
            pr=np.array([landmarks_R[i].x, landmarks_R[i].y, landmarks_R[i].z])
            inter_distances.append(np.linalg.norm(pl-pr)/avg_scale)

        # 2 : angles intermains

        def get_normal(lm):
            p0=np.array([lm[0].x, lm[0].y, lm[0].z])
            p5=np.array([lm[5].x, lm[5].y, lm[5].z])
            p17=np.array([lm[17].x, lm[17].y, lm[17].z])
            vect1=p0-p5
            vect2=p17-p0
            res=np.cross(vect1, vect2)
            return res/(np.linalg.norm(res)+1e-6)
        
        angle_palms=np.arccos(np.clip(np.dot(get_normal(landmarks_L), get_normal(landmarks_R)), -1.0, 1.0))

        return inter_distances + [angle_palms]


# --- GESTION DU FICHIER CSV ---

def init_csv():
    """Crée le fichier avec l'en-tête s'il n'existe pas"""
    if not os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, mode='w', newline='') as f:
            header = [f'L_{i}' for i in range(NUM_FEATURES_PER_HAND)] + \
                     [f'R_{i}' for i in range(NUM_FEATURES_PER_HAND)] + \
                     [f'Inter_{i}' for i in range(6)]
            csv.writer(f).writerow(header)

# --- BOUCLE PRINCIPALE ---
def main():
    init_csv()
    extractor = FeatureExtractor()
    recording = False
    
    # Configuration MediaPipe pour 2 mains
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.8
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)

    print("=== SYSTÈME D'ACQUISITION JJK PRÊT ===")
    print(f"Cible : {DATASET_PATH}")
    print("Commandes : [G] Toggle Recording | [Q] Quitter")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1) # Effet miroir
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = detector.detect(mp_image)

        feat_L, feat_R = [-10.0]*88, [-10.0]*88
        lm_L, lm_R = None, None
        sc_L, sc_R = 0, 0

        color= (0, 255, 0)

        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                # Détection du côté (Left/Right)
                side = result.handedness[i][0].category_name 
                f, s = extractor.get_hand_features(landmarks)
                if side == 'Left':
                    feat_L, lm_L, sc_L = f, landmarks, s
                else:
                    feat_R, lm_R, sc_R = f, landmarks, s

                # Dessin du squelette simplifié
                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, color, -1)

            # Calcul des features intermains
            feat_inter = extractor.get_inter_hand_features(lm_L, lm_R, sc_L, sc_R)
            
            # Assemblage du vecteur final (182)
            full_vector = feat_L + feat_R + feat_inter

            # ENREGISTREMENT
            if recording:
                with open(DATASET_PATH, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(full_vector)

        # Interface Utilisateur
        rec_color = (0, 0, 255) if recording else (0, 255, 0)
        status = "● ENREGISTREMENT EN COURS" if recording else "EN ATTENTE (G)"
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)
        cv2.imshow("Acquisition Dataset JJK", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            recording = not recording
            print(f"Statut Recording : {recording}")
        elif key == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()