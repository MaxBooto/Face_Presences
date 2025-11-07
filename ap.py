from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for, send_file
import cv2
import os
import shutil
import numpy as np
from datetime import datetime
import mysql.connector
import hashlib
import pandas as pd
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'votre_cle_secrete_statique'

MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'jeremie',
    'raise_on_warnings': True,
    'consume_results': True
}

IMAGE_COUNT = 50
CONFIDENCE_THRESHOLD = 4000
FACE_SIZE = (200, 200)
camera = cv2.VideoCapture(0)
neural_model = None
label_map = None
face_detector = None

# Fonctions utilitaires
def hash_password_md5(password):
    return hashlib.md5(password.encode('utf-8')).hexdigest()

def init_neural_network():
    global neural_model, label_map, face_detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_detector.empty():
        print("Erreur : Impossible de charger le modèle de détection de visages pour le réseau de neurones.")
        raise Exception("Impossible de charger le modèle de détection de visages.")
    try:
        if os.path.exists("neural_network_model.xml") and os.path.exists("label_map.npy"):
            neural_model = cv2.face.EigenFaceRecognizer_create()
            neural_model.read("neural_network_model.xml")
            label_map = np.load("label_map.npy", allow_pickle=True).item()
            print("Réseau de neurones chargé avec succès.")
        else:
            neural_model = None
            label_map = {}
            print("Aucun modèle de réseau de neurones trouvé. Veuillez entraîner le modèle via la route /train_neural_network.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle neuronal : {e}")
        neural_model = None
        label_map = {}

def get_today():
    return datetime.now().strftime("%d/%m/%Y")

def get_time():
    return datetime.now().strftime("%H:%M:%S")

def add_presence(name):
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        c = conn.cursor()
        today = get_today()
        c.execute("SELECT p.id FROM presences p JOIN users u ON p.id_users = u.id WHERE u.nom=%s AND p.date=%s", (name, today))
        existing_presence = c.fetchone()
        if existing_presence:
            print(f"Présence déjà enregistrée pour {name} le {today} via le réseau de neurones")
            conn.close()
            return get_presences_list(today)
        time = get_time()
        c.execute("SELECT id FROM users WHERE nom=%s", (name,))
        user_id = c.fetchone()
        if user_id:
            c.execute("INSERT INTO presences (date, heure, id_users) VALUES (%s, %s, %s)",
                     (today, time, user_id[0]))
            conn.commit()
            print(f"Présence ajoutée pour {name} le {today} à {time} via inférence neuronale")
        else:
            print(f"Utilisateur {name} non trouvé dans la base")
        conn.close()
        return get_presences_list(today)
    except mysql.connector.Error as e:
        print(f"Erreur lors de l'ajout de la présence : {e}")
        return []

def get_presences_list(date):
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        c = conn.cursor()
        c.execute("SELECT p.date, p.heure, u.nom FROM presences p JOIN users u ON p.id_users = u.id WHERE p.date=%s ORDER BY p.heure", (date,))
        presences = [{"date": row[0], "heure": row[1], "nom": row[2]} for row in c.fetchall()]
        conn.close()
        print(f"Présences récupérées pour {date} : {len(presences)} enregistrements")
        return presences
    except mysql.connector.Error as e:
        print(f"Erreur lors de la récupération des présences : {e}")
        return []

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print("Erreur : Impossible de lire la frame de la caméra pour l'inférence neuronale.")
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            recognized_names = []
            if neural_model is not None:
                for (x, y, w, h) in faces:
                    face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
                    label, confidence = neural_model.predict(face)
                    print(f"Confiance neuronale : {confidence:.2f} (Seuil : {CONFIDENCE_THRESHOLD})")
                    if confidence < CONFIDENCE_THRESHOLD:
                        recognized_name = label_map.get(label, "INCONNU")
                        if recognized_name != "INCONNU":
                            recognized_names.append(recognized_name)
                        cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, "INCONNU", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                for (x, y, w, h) in faces:
                    cv2.putText(frame, "INCONNU", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if recognized_names:
                add_presence(recognized_names[0])
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def load_images(folder):
    images = []
    labels = []
    label_id = 0
    label_map = {}
    print(f"Chargement des données pour l'entraînement du réseau de neurones depuis : {folder}")
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            label_map[label_id] = subdir
            print(f"Exploration du dossier pour l'entraînement neuronal : {subdir_path}")
            image_files = [f for f in os.listdir(subdir_path) if f.endswith('.jpg')]
            print(f"Images trouvées dans {subdir} : {len(image_files)} fichiers")
            for filename in image_files:
                img_path = os.path.join(subdir_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Image invalide ignorée : {img_path}")
                    continue
                try:
                    img = cv2.resize(img, FACE_SIZE)
                    images.append(img)
                    labels.append(label_id)
                    print(f"Image chargée pour l'entraînement : {img_path}")
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {img_path} : {e}")
            label_id += 1
    if not images:
        print("Aucune image valide trouvée pour l'entraînement du réseau de neurones.")
    print(f"Total images chargées : {len(images)}, Total étiquettes : {len(labels)}")
    return images, np.array(labels), label_map

# Routes
@app.route('/create_admin', methods=['GET', 'POST'])
def create_admin():
    if request.method == 'POST':
        nom = request.form['nom']
        prenom = request.form['prenom']
        email = request.form['email']
        password = request.form['password']
        hashed_password = hash_password_md5(password)
        try:
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            c = conn.cursor()
            c.execute("INSERT INTO admin (nom, prenom, mail, password) VALUES (%s, %s, %s, %s)",
                     (nom, prenom, email, hashed_password))
            conn.commit()
            conn.close()
            print(f"Administrateur créé : {nom} {prenom} ({email})")
            return render_template('create_admin.html', message="Administrateur créé avec succès")
        except mysql.connector.Error as e:
            print(f"Erreur lors de la création de l'admin : {e}")
            return render_template('create_admin.html', error=f"Erreur : {e}")
    return render_template('create_admin.html')

@app.route('/')
@app.route('/dashboard')
def index():
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = hash_password_md5(password)
        try:
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            c = conn.cursor()
            c.execute("SELECT id, password FROM admin WHERE mail=%s", (email,))
            admin = c.fetchone()
            conn.close()
            if admin and admin[1] == hashed_password:
                session['admin_id'] = admin[0]
                print(f"Connexion réussie pour admin ID : {admin[0]}")
                return redirect(url_for('index'))
            print(f"Échec de la connexion : Email ou mot de passe incorrect pour {email}")
            return render_template('login.html', error="Email ou mot de passe incorrect")
        except mysql.connector.Error as e:
            print(f"Erreur de connexion à la base de données : {e}")
            return render_template('login.html', error=f"Erreur de connexion à la base de données : {e}")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin_id', None)
    print("Déconnexion effectuée")
    return redirect(url_for('login'))

@app.route('/recognition')
def recognition():
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    return render_template('recognition.html')

@app.route('/video_feed')
def video_feed():
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/presence_list', methods=['GET'])
def get_presences():
    if 'admin_id' not in session:
        print("Erreur : Accès non autorisé à /presence_list")
        return jsonify(success=False, message="Non autorisé")
    today = get_today()
    presences = get_presences_list(today)
    print(f"Présences récupérées pour {today} : {len(presences)} enregistrements via le réseau de neurones")
    return jsonify(presences)

@app.route('/clear_presence', methods=['DELETE'])
def clear_presences():
    if 'admin_id' not in session:
        print("Erreur : Accès non autorisé à /clear_presence")
        return jsonify(success=False, message="Non autorisé")
    try:
        today = get_today()
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        c = conn.cursor()
        c.execute("DELETE FROM presences WHERE date=%s", (today,))
        conn.commit()
        conn.close()
        print(f"Présences vidées pour {today}")
        return jsonify(success=True, message="Liste des présences vidée.")
    except mysql.connector.Error as e:
        print(f"Erreur lors du vidage des présences : {e}")
        return jsonify(success=False, message=f"Erreur : {e}")

@app.route('/history', methods=['GET'])
def get_presences_by_date():
    if 'admin_id' not in session:
        print("Erreur : Accès non autorisé à /history")
        return jsonify(success=False, message="Non autorisé")
    try:
        selected_date = request.args.get('date')
        if not selected_date:
            print("Erreur : Date non fournie")
            return jsonify(success=False, message="Date requise")
        presences = get_presences_list(selected_date)
        print(f"Présences récupérées pour {selected_date} : {len(presences)} enregistrements via le réseau de neurones")
        return jsonify(presences)
    except mysql.connector.Error as e:
        print(f"Erreur lors de la récupération des présences : {e}")
        return jsonify(success=False, message=f"Erreur de base de données : {e}")
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        return jsonify(success=False, message=f"Erreur : {e}")

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if 'admin_id' not in session:
        print("Erreur : Accès non autorisé à /add_user")
        return redirect(url_for('login'))
    if request.method == 'POST':
        try:
            data = request.get_json()
            action = data.get('action')
            print(f"Requête POST reçue pour /add_user avec action : {action}")
            nom = data.get('nom', '').strip()
            prenom = data.get('prenom', '').strip()
            sexe = data.get('sexe', '')
            print(f"Données reçues : nom={nom}, prenom={prenom}, sexe={sexe}")
            
            if not nom or not prenom or not sexe:
                print(f"Erreur : Champs manquants - nom: {nom}, prenom: {prenom}, sexe: {sexe}")
                return jsonify(success=False, message="Tous les champs (nom, prénom, sexe) sont requis.")
            
            if action == 'capture':
                save_folder = f"dataset/{nom}"
                print(f"Création du dossier pour les données neuronales : {save_folder}")
                os.makedirs(save_folder, exist_ok=True)
                face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                if face_detector.empty():
                    print("Erreur : Impossible de charger le modèle de détection de visages pour le réseau de neurones.")
                    return jsonify(success=False, message="Erreur : Impossible de charger le modèle de détection de visages.")
                print("Modèle de détection de visages chargé pour la capture neuronale")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Erreur : Impossible d'accéder à la webcam.")
                    return jsonify(success=False, message="Erreur : Impossible d'accéder à la webcam.")
                print("Webcam ouverte avec succès pour la capture")
                
                try:
                    count = 0
                    while count < IMAGE_COUNT:
                        ret, frame = cap.read()
                        if not ret:
                            print("Erreur : Impossible de capturer l'image pour l'entraînement neuronal.")
                            raise Exception("Impossible de capturer l'image.")
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
                        print(f"Visages détectés : {len(faces)}")
                        if len(faces) == 0:
                            print("Aucun visage détecté dans cette frame")
                        for (x, y, w, h) in faces:
                            face = gray[y:y+h, x:x+w]
                            face = cv2.resize(face, FACE_SIZE)
                            image_path = os.path.join(save_folder, f"{nom}_{count + 1}.jpg")
                            if cv2.imwrite(image_path, face):
                                print(f"Image {count + 1} enregistrée pour l'entraînement neuronal : {image_path}")
                                count += 1
                            else:
                                print(f"Erreur : Échec de l'écriture de l'image {image_path}")
                        cv2.imshow("Capture en cours pour réseau neuronal", frame)
                        if cv2.waitKey(100) == ord('q'):
                            print("Capture interrompue par l'utilisateur (touche 'q')")
                            break
                    print(f"Capture terminée pour {nom} : {count} images pour l'entraînement neuronal")
                    if count < IMAGE_COUNT:
                        return jsonify(success=False, message=f"Capture incomplète : seulement {count} images capturées sur {IMAGE_COUNT}.")
                    return jsonify(success=True, message=f"Capture terminée. {count} images enregistrées pour le réseau neuronal.")
                except Exception as e:
                    print(f"Erreur lors de la capture : {str(e)}")
                    return jsonify(success=False, message=f"Erreur lors de la capture : {str(e)}")
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Webcam libérée et fenêtres fermées")
            
            elif action == 'save':
                conn = mysql.connector.connect(**MYSQL_CONFIG)
                c = conn.cursor(buffered=True)
                print("Connexion à la base de données établie")
                
                c.execute("SELECT id FROM users WHERE nom=%s AND prenom=%s AND sexe=%s", (nom, prenom, sexe))
                user = c.fetchone()
                print(f"Vérification utilisateur : {user}")
                
                if user:
                    conn.close()
                    print(f"Utilisateur existe déjà : {nom} {prenom} ({sexe})")
                    return jsonify(success=False, message=f"Utilisateur {nom} {prenom} ({sexe}) existe déjà dans la base.")
                
                print(f"Tentative d'insertion de l'utilisateur : {nom} {prenom} ({sexe})")
                c.execute("INSERT INTO users (nom, prenom, sexe) VALUES (%s, %s, %s)", (nom, prenom, sexe))
                conn.commit()
                print("Utilisateur inséré dans la table users")
                
                c.execute("SELECT id FROM users WHERE nom=%s AND prenom=%s AND sexe=%s", (nom, prenom, sexe))
                user_id = c.fetchone()
                c.fetchall()
                if not user_id:
                    conn.close()
                    print(f"Erreur : Impossible de récupérer l'ID de l'utilisateur {nom} {prenom}")
                    return jsonify(success=False, message="Erreur : Impossible de récupérer l'ID de l'utilisateur après insertion.")
                user_id = user_id[0]
                
                temp_folder = f"dataset/{nom}"
                save_folder = f"dataset/{nom}"
                os.makedirs(save_folder, exist_ok=True)
                
                if not os.path.exists(temp_folder):
                    conn.close()
                    print(f"Erreur : Dossier temporaire {temp_folder} n'existe pas")
                    return jsonify(success=False, message="Erreur : Aucune image capturée trouvée.")
                
                image_paths = []
                for filename in os.listdir(temp_folder):
                    src_path = os.path.join(temp_folder, filename)
                    dst_path = os.path.join(save_folder, filename)
                    shutil.move(src_path, dst_path)
                    image_paths.append(dst_path)
                    c.execute("INSERT INTO image (photo, id_users) VALUES (%s, %s)", (dst_path, user_id))
                    print(f"Image enregistrée dans la base pour l'entraînement neuronal : {dst_path}")
                
                conn.commit()
                conn.close()
                
                shutil.rmtree(temp_folder, ignore_errors=True)
                
                print(f"Utilisateur et images enregistrés : {nom} {prenom} ({sexe}), ID={user_id}, {len(image_paths)} images")
                return jsonify(success=True, message=f"Utilisateur {nom} {prenom} et {len(image_paths)} images enregistrés avec succès pour le réseau neuronal.")
            
        except mysql.connector.Error as e:
            print(f"Erreur de base de données : {e}")
            return jsonify(success=False, message=f"Erreur de base de données : {str(e)}")
        except Exception as e:
            print(f"Erreur inattendue : {e}")
            return jsonify(success=False, message=f"Erreur inattendue : {str(e)}")
    print("Rendu de la page add_user.html")
    return render_template('add_user.html')

@app.route('/train_neural_network', methods=['POST'])
def train_neural_network():
    if 'admin_id' not in session:
        print("Erreur : Accès non autorisé à la route /train_neural_network")
        return jsonify(success=False, message="Non autorisé")
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        print("Erreur : Le dossier 'dataset' n'existe pas pour l'entraînement neuronal.")
        return jsonify(success=False, message="Erreur : Le dossier 'dataset' n'existe pas.")
    images, labels, label_map = load_images(dataset_path)
    print(f"Images chargées : {len(images)}, Étiquettes : {len(labels)}, Label Map : {label_map}")
    if len(images) == 0 or len(labels) == 0:
        print("Erreur : Aucune image valide trouvée pour l'entraînement du réseau de neurones.")
        return jsonify(success=False, message="Erreur : Aucune image valide trouvée dans 'dataset'.")
    if len(images) != len(labels):
        print("Erreur : Le nombre d'images et d'étiquettes ne correspond pas.")
        return jsonify(success=False, message="Erreur : Le nombre d'images et d'étiquettes ne correspond pas.")
    if len(images) < IMAGE_COUNT:
        print(f"Erreur : Nombre d'images insuffisant ({len(images)} trouvées, {IMAGE_COUNT} requises).")
        return jsonify(success=False, message=f"Erreur : Nombre d'images insuffisant ({len(images)} trouvées, {IMAGE_COUNT} requises).")
    try:
        neural_model = cv2.face.EigenFaceRecognizer_create()
        print("Entraînement du réseau de neurones avec les images...")
        neural_model.train(images, np.array(labels))
        neural_model.save("neural_network_model.xml")
        np.save("label_map.npy", label_map)
        init_neural_network()
        print("Réseau de neurones entraîné et sauvegardé.")
        return jsonify(success=True, message="Réseau de neurones entraîné et sauvegardé.")
    except Exception as e:
        print(f"Erreur lors de l'entraînement du réseau neuronal : {str(e)}")
        return jsonify(success=False, message=f"Erreur lors de l'entraînement : {str(e)}")

@app.route('/history')
def history():
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    return render_template('history.html')

@app.route('/manage_users')
def manage_users():
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        c = conn.cursor()
        c.execute("SELECT id, nom, prenom, sexe FROM users")
        users = [{"id": row[0], "nom": row[1], "prenom": row[2], "sexe": row[3]} for row in c.fetchall()]
        conn.close()
        print(f"Utilisateurs récupérés : {len(users)}")
        return render_template('manage_users.html', users=users)
    except mysql.connector.Error as e:
        print(f"Erreur lors de la récupération des utilisateurs : {e}")
        return render_template('manage_users.html', error=f"Erreur : {e}")

@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    c = conn.cursor()
    if request.method == 'POST':
        data = request.get_json()
        nom = data['nom']
        prenom = data['prenom']
        sexe = data['sexe']
        try:
            c.execute("UPDATE users SET nom=%s, prenom=%s, sexe=%s WHERE id=%s", (nom, prenom, sexe, user_id))
            conn.commit()
            conn.close()
            print(f"Utilisateur modifié : ID={user_id}, nom={nom}, prenom={prenom}, sexe={sexe}")
            return jsonify(success=True, message="Utilisateur modifié avec succès")
        except mysql.connector.Error as e:
            conn.close()
            print(f"Erreur lors de la modification de l'utilisateur : {e}")
            return jsonify(success=False, message=f"Erreur : {e}")
    c.execute("SELECT nom, prenom, sexe FROM users WHERE id=%s", (user_id,))
    user = c.fetchone()
    conn.close()
    if user:
        print(f"Rendu de la page edit_user.html pour ID={user_id}")
        return render_template('edit_user.html', user={"id": user_id, "nom": user[0], "prenom": user[1], "sexe": user[2]})
    print(f"Utilisateur non trouvé : ID={user_id}")
    return redirect(url_for('manage_users'))

@app.route('/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    if 'admin_id' not in session:
        print("Erreur : Accès non autorisé à /delete_user")
        return jsonify(success=False, message="Non autorisé")
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        c = conn.cursor()
        
        c.execute("SELECT nom FROM users WHERE id=%s", (user_id,))
        user = c.fetchone()
        if not user:
            conn.close()
            print(f"Utilisateur ID={user_id} non trouvé")
            return jsonify(success=False, message="Utilisateur non trouvé")
        user_nom = user[0]
        print(f"Début de la suppression pour l'utilisateur ID={user_id}, nom={user_nom}")
        
        c.execute("DELETE FROM presences WHERE id_users=%s", (user_id,))
        presences_deleted = c.rowcount
        print(f"{presences_deleted} enregistrements supprimés de la table presences")
        
        c.execute("DELETE FROM image WHERE id_users=%s", (user_id,))
        images_deleted = c.rowcount
        print(f"{images_deleted} enregistrements supprimés de la table image")
        
        c.execute("DELETE FROM users WHERE id=%s", (user_id,))
        users_deleted = c.rowcount
        print(f"{users_deleted} utilisateur supprimé de la table users")
        
        conn.commit()
        conn.close()
        
        dataset_folder = f"dataset/{user_nom}"
        if os.path.exists(dataset_folder):
            shutil.rmtree(dataset_folder, ignore_errors=True)
            print(f"Dossier supprimé : {dataset_folder}")
        else:
            print(f"Dossier non trouvé, ignoré : {dataset_folder}")
        
        print(f"Suppression terminée pour l'utilisateur ID={user_id}, nom={user_nom}")
        return jsonify(success=True, message=f"Utilisateur {user_nom} et toutes ses données supprimés avec succès")
    except mysql.connector.Error as e:
        print(f"Erreur lors de la suppression de l'utilisateur : {e}")
        return jsonify(success=False, message=f"Erreur de base de données : {str(e)}")
    except Exception as e:
        print(f"Erreur inattendue lors de la suppression : {e}")
        return jsonify(success=False, message=f"Erreur : {str(e)}")

@app.route('/export_users')
def export_users_excel():
    if 'admin_id' not in session:
        print("Erreur : Accès non autorisé à /export_users")
        return jsonify(success=False, message="Non autorisé"), 401
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        c = conn.cursor()
        c.execute("SELECT nom, prenom, sexe FROM users")
        users = [{"Nom": row[0], "Prénom": row[1], "Sexe": row[2]} for row in c.fetchall()]
        conn.close()
        
        if not users:
            print("Aucun utilisateur à exporter")
            return jsonify(success=False, message="Aucun utilisateur à exporter"), 400
        
        today = datetime.now().strftime("%d-%m-%Y")
        df = pd.DataFrame(users)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet('Utilisateurs')
            worksheet.write(0, 0, f"Liste des Utilisateurs au {today}")
            df.to_excel(writer, sheet_name='Utilisateurs', startrow=2, index=False)
            worksheet.set_column('A:A', 20)
            worksheet.set_column('B:B', 20)
            worksheet.set_column('C:C', 10)
        
        output.seek(0)
        file_name = f"utilisateurs_{today}.xlsx"
        print(f"Exportation des utilisateurs réussie : {file_name}")
        return send_file(
            output,
            download_name=file_name,
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except mysql.connector.Error as e:
        print(f"Erreur de base de données lors de l'exportation des utilisateurs : {e}")
        return jsonify(success=False, message=f"Erreur de base de données : {str(e)}"), 500
    except ImportError as e:
        print(f"Erreur d'importation de module : {e}")
        return jsonify(success=False, message="Erreur serveur : dépendances manquantes"), 500
    except Exception as e:
        print(f"Erreur inattendue lors de l'exportation des utilisateurs : {e}")
        return jsonify(success=False, message=f"Erreur serveur : {str(e)}"), 500

@app.route('/export_presence', methods=['POST'])
def export_presences_excel():
    if 'admin_id' not in session:
        print("Erreur : Accès non autorisé à /export_presence")
        return jsonify(success=False, message="Non autorisé"), 401
    try:
        data = request.get_json()
        selected_date = data.get('date')
        print(f"Requête d'exportation reçue pour la date : {selected_date}")
        if not selected_date:
            print("Erreur : Date non fournie")
            return jsonify(success=False, message="Date requise"), 400
        
        presences = get_presences_list(selected_date)
        if not presences:
            print(f"Aucune présence à exporter pour {selected_date}")
            return jsonify(success=False, message="Aucune présence à exporter pour cette date"), 400
        
        print(f"Préparation du fichier Excel avec {len(presences)} présences")
        df = pd.DataFrame(presences, columns=['date', 'heure', 'nom'])
        df.columns = ['Date', 'Heure', 'Nom']
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet('Présences')
            worksheet.write(0, 0, f"Liste des Présences du {selected_date}")
            df.to_excel(writer, sheet_name='Présences', startrow=3, index=False)
            worksheet.set_column('A:A', 15)
            worksheet.set_column('B:B', 10)
            worksheet.set_column('C:C', 20)
        
        output.seek(0)
        file_name = f"presences_{selected_date.replace('/', '-')}.xlsx"
        print(f"Exportation réussie : {file_name}")
        return send_file(
            output,
            download_name=file_name,
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except mysql.connector.Error as e:
        print(f"Erreur de base de données lors de l'exportation des présences : {e}")
        return jsonify(success=False, message=f"Erreur de base de données : {str(e)}"), 500
    except ImportError:
        print(f"Erreur d'importation de module : {e}")
        return jsonify(success=False, message=f"Erreur serveur : dépendances manquantes"), 500
    except Exception as e:
        print(f"Erreur lors de l'exportation des présences : {e}")
        return jsonify(success=False, message=f"Erreur : {str(e)}")

@app.route('/export_history', methods=['POST'])
def export_history():
    return export_presences_excel()

if __name__ == '__main__':
    init_neural_network()
    app.run(host='0.0.0.0', port=5000, debug=True)