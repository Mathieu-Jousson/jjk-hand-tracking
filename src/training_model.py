import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR=os.path.join(ROOT_DIR, "datasets")


# --- 1. CHARGEMENT ET FUSION DES DONNÉES ---
def load_and_prepare_data():
    classes = {
        'idle': 0,
        'gojo': 1,
        'sukuna': 2,
        'mahito': 3,
        'megumi': 4
    }
    
    all_data = []
    for name, label in classes.items():
        path = os.path.join(DATASETS_DIR, f'{name}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['target'] = label
            all_data.append(df)
            print(f"Chargé : {name} ({len(df)} samples)")
    
    full_df = pd.concat(all_data, ignore_index=True)
    return full_df

# --- 2. ENTRAÎNEMENT ---
def train():
    print("Préparation des données...")
    df = load_and_prepare_data()
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split 80% Entraînement / 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Entraînement de la forêt (182 features)...")
    model = RandomForestClassifier(
    n_estimators=200,      
    max_depth=20,         
    min_samples_split=10,  
    n_jobs=-1, 
    random_state=42
)
    
    model.fit(X_train, y_train)
    
    # --- 3. ÉVALUATION ---
    score = model.score(X_test, y_test)
    print(f"\nPrécision globale (Accuracy) : {score:.2%}")
    
    y_pred = model.predict(X_test)
    print("\nTableau de performance :")
    print(classification_report(y_test, y_pred))
    
    # --- 4. SAUVEGARDE ---
    timestamp = time.strftime("%Y%m%d-%H%M")
    model_name = f"jjk_model_{timestamp}.joblib"
    joblib.dump(model, model_name)
    print(f"\nModèle sauvegardé sous : {model_name}")

if __name__ == "__main__":
    train()