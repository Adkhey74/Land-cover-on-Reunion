from models.create_model import model_creation, get_global_variable_progress_model_creation
from flask import Flask, render_template, request, redirect, url_for,json, flash, session,jsonify,Response,stream_with_context, make_response
from models.inference import inference_processus, get_global_variable_progress
import os
from PIL import Image
import io
import base64
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from tqdm import tqdm

from extensions import db
from model import Utilisateur, CheminModeleIA, ImageTIF800, ImageTIF1600,CheminImageCouleur
from azure.storage.blob import BlobClient
from io import BytesIO
from coordinate import getCoordinate
import random
from multiprocessing import Process
from threading import Thread
from queue import Queue
app = Flask(__name__)

database_uri = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/mydatabase')
app.config['SQLALCHEMY_DATABASE_URI'] = database_uri #'postgresql://postgres:postgres@host.docker.internal/ia_reunion'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.secret_key = "samuel"
CORS(app)

db.init_app(app)



# Page d'accueil
@app.route('/')
def home():
    if 'connection' not in session:
        return redirect(url_for('connexion_compte'))
    return render_template('acceuil.html',active_page='acceuil.html')

# Page "Dataset"
@app.route('/dataset.html', methods=['GET', 'POST'])
def dataset():
    if 'connection' not in session:
        return redirect(url_for('connexion_compte'))
    
    if request.method == 'POST':

        image_height = request.form.get('image_height')
        image_width = request.form.get('image_width')
        f_height = request.form.get('f_height')
        f_width = request.form.get('f_width')
        pixel_resoltion = request.form.get('pixel_resoltion')
        tile_ratio = request.form.get('tile_ratio')
        filter_size=request.form.get('filter_size')
        pool_size=request.form.get('pool_size')


        return redirect(url_for('create_model',
            image_height=image_height,
            image_width=image_width,
            f_height=f_height,
            f_width=f_width,
            pixel_resoltion=pixel_resoltion,
            tile_ratio=tile_ratio,
            filter_size=filter_size,
            pool_size=pool_size
        ))

    return render_template('dataset.html', active_page='dataset.html')

# Page "Model"
@app.route('/create_model.html', methods=['GET', 'POST'])
def create_model():
    if 'connection' not in session:
        return redirect(url_for('connexion_compte'))

    # Récupérez les valeurs passées en tant que paramètres
    image_height = int(request.args.get('image_height'))
    image_width = int(request.args.get('image_width'))
    f_height = int(request.args.get('f_height'))
    f_width = int(request.args.get('f_width'))
    pixel_resoltion = int(request.args.get('pixel_resoltion'))
    tile_ratio = float(request.args.get('tile_ratio'))
    filter_size=int(request.args.get('filter_size'))
    pool_size=int(request.args.get('pool_size'))

    if image_height == None or image_width == None or f_height == None or f_width == None or pixel_resoltion == None or tile_ratio == None or filter_size == None or pool_size == None :
        return redirect(url_for('dataset'))

    # Créez un dictionnaire pour stocker les valeurs existantes
    values = {
        'image_height': image_height,
        'image_width': image_width,
        'f_height':f_height,
        'f_width': f_width,
        'pixel_resoltion': pixel_resoltion,
        'tile_ratio': tile_ratio,
        'filter_size':filter_size,
        'pool_size':pool_size
    }

    if request.method == 'POST':
        nombre_neurone = []
        nombre_couche = int(request.form.get('nombre_couche'))
        activation_function = request.form.get('activation_function')
        optimizer = request.form.get('optimizer')
        loss = request.form.get('loss')
        metrics = request.form.get('metrics')
        for i in range(1, nombre_couche + 1):
            champ_neurone = 'neurone_couche' + str(i)
            nombre_neurone.append(int(request.form.get(champ_neurone)))
            values.update({
                'nombre_couche': nombre_couche,
                'activation_function': activation_function,
                'optimizer': optimizer,
                'loss': loss,
                'metrics': metrics,
                'nombre_neurone': nombre_neurone
            })
        
        # model_creation(image_height,image_width,pixel_resoltion,f_height,f_width,tile_ratio,nombre_couche,nombre_neurone,filter_size,pool_size,activation_function,optimizer,loss,metrics)
        return redirect(url_for('training',values=values))

    return render_template('create_model.html',active_page='create_model.html',values=values)

@app.route('/training.html',methods=['GET', 'POST'])
def training():
    if 'connection' not in session:
        return redirect(url_for('connexion_compte'))

    if request.method == 'POST':
        values = request.args.get('values')

        if values == None :
            return redirect(url_for('dataset'))

        values = values.replace("'", "\"")
        values_dict = json.loads(values)
        
        # Récupérer les valeurs spécifiques par clé
        image_height = int(values_dict.get('image_height'))
        image_width = int(values_dict.get('image_width'))
        pixel_resoltion = values_dict.get('pixel_resoltion')
        f_height = int(values_dict.get('f_height'))
        f_width = int(values_dict.get('f_width'))
        tile_ratio = int(values_dict.get('tile_ratio'))
        nombre_couche = int(values_dict.get('nombre_couche'))
        nombre_neurone = values_dict.get('nombre_neurone', [])
        filter_size = int(values_dict.get('filter_size'))
        pool_size = int(values_dict.get('pool_size'))
        activation_function = values_dict.get('activation_function')
        optimizer = values_dict.get('optimizer')
        loss = values_dict.get('loss')
        metrics = values_dict.get('metrics')


        model_name = request.form.get('model_name')
        description = request.form.get('model_description')
        patience_early_stopping = int(request.form.get('patience_early_stopping'))
        restore_best_weights = bool(request.form.get('restore_best_weights'))
        min_delta_early_stopping = float(request.form.get('min_delta_early_stopping'))
        mode = request.form.get('mode')
        factor = float(request.form.get('factor'))
        patience_reduce_lr= int(request.form.get('patience_reduce_lr'))
        min_lr_reduce_lr=float(request.form.get('min_lr_reduce_lr'))
        min_delta= float(request.form.get('min_delta'))
        epochs= int(request.form.get('epochs'))




        model_creation(image_height, image_width, pixel_resoltion, f_height, f_width, tile_ratio,
                    nombre_couche, nombre_neurone, filter_size, pool_size, activation_function,
                    optimizer, loss, metrics,patience_early_stopping,restore_best_weights,min_delta_early_stopping,mode,factor,patience_reduce_lr,min_lr_reduce_lr,min_delta,epochs,model_name)
        
        newModel = CheminModeleIA(
            id_utilisateur = session['user']['id'],
            chemin = model_name + '.h5',
            description = description,
            image_height = image_height,
            image_width = image_width,
            pixel_resolution = pixel_resoltion,
            f_height = f_height,
            f_width = f_width,
            tile_ratio = tile_ratio,
            nombre_couche = nombre_couche,
            nombre_neurone = nombre_neurone,
            filter_size = filter_size,
            pool_size = pool_size,
            activation_function = activation_function,
            optimizer = optimizer,
            loss = loss,
            metrics = metrics,
            patience_early_stopping = patience_early_stopping,
            restore_best_weights = restore_best_weights,
            min_delta_early_stopping = min_delta_early_stopping,
            mode = mode,
            factor = factor,
            patience_reduce_lr = patience_reduce_lr,
            min_lr_reduce_lr = min_lr_reduce_lr,
            min_delta = min_delta,
            epochs = epochs,
            model_name = model_name,
        )

        db.session.add(newModel)
        db.session.commit()
        flash("Model créé avec succès!", "success")

        return redirect(url_for('model',values=values_dict))

    return render_template('training.html', active_page='training.html')


# Page "Model"
@app.route('/model.html')
def model():
    if 'connection' not in session:
        return redirect(url_for('connexion_compte'))

    # # Récupérez les valeurs passées en tant que paramètres
    # img_height = request.args.get('img_height')
    # img_width = request.args.get('img_width')
    # frame_height = request.args.get('frame_height')
    # frame_width = request.args.get('frame_width')
    # pixel_res = request.args.get('pixel_res')
    # tile_ratio = request.args.get('tile_ratio')
    # values = request.args.get('values')
    # values = values.replace("'", "\"")
    # values_dict = json.loads(values)

        # Créez un dictionnaire pour stocker les valeurs existantes

    # ... le reste de votre code pour récupérer les autres valeurs ...

    dossier_modeles = '/app/static/IA/model'
    id_utilisateur = session['user']['id']

    model1 = CheminModeleIA(
            id_utilisateur = id_utilisateur,
            chemin = "classic_model.h5",
            description = "This Sequential Convolutional Neural Network (CNN) model is designed for image classification. It processes images of 25x25 pixels with 4 channels. The model consists of three convolutional layers, each followed by a MaxPooling2D layer, with increasing filter sizes (32, 64, 128) and 'relu' activation. After convolution, the data is flattened and passes through a dense layer with 128 neurons (also 'relu' activated), followed by a softmax output layer with 27 neurons, suitable for multi-class classification.",
            model_name = "classic_model",
        )
    model2 = CheminModeleIA(
            id_utilisateur = id_utilisateur,
            chemin = "successive_model.h5",
            description = "This Convolutional Neural Network (CNN) model, named Successive Model, is designed for hierarchical and dependent image classification. It processes images of size 25x25 pixels with 4 channels. The model features three output layers, each predicting different class categories with a dependency chain: the prediction of the first layer (4 classes) influences the second (11 classes), and the second affects the third (27 classes). Compiled with the 'adam' optimizer and using 'categorical_crossentropy' for loss, it's ideal for complex, layered classification tasks.",
            model_name = "successive_model",
        )
    list_of_model_files = [model1, model2]

    modelsByUser = db.session.query(CheminModeleIA).filter_by(id_utilisateur=id_utilisateur).all()

    list_of_model_files.extend(modelsByUser)
    # test
    # Affichez ces valeurs dans le modèle HTML
    return render_template('model.html', active_page='model.html', list_of_model_files=list_of_model_files)    # Créez un dictionnaire pour stocker les valeurs existantes



@app.route('/images.html')
def images():
    if 'connection' not in session:
        return redirect(url_for('connexion_compte'))
    
    model = request.args.get('model')
    if model == None :
        return redirect(url_for('model'))
    # Nombre d'images à récupérer
    nombre_images = 8
    tiff_path = '/app/static/IA/data/sat/test/extract.tif'  # Assurez-vous de spécifier le bon chemin
    image_extract = get_encoded_image(tiff_path)
    image_paths = []

    if 'image_start' not in session:
    # Initialiser la variable de session à 1
        session['image_start'] = 1    # Boucle pour récupérer les quatre premières images
    print(session['image_start'])

    # Envoyez la liste des chemins d'images au modèle
    return render_template('images.html', active_page='images.html', image=image_paths, model=model,image_extract=image_extract,tiff_path=tiff_path)


@app.route('/update_images')
def update_images():

    dimension_img =  request.args.get('dimension_img', default='pas de num')
    print("dimension: "+dimension_img)

        
    nombre_images = 8

    def generate_images(dimension_img):
        for numero_image in range(session['image_start'], nombre_images + session['image_start']):
            if(dimension_img == '1600'):
                url_complete_with_sas = ImageTIF1600.query.get(numero_image)
            elif(dimension_img == '800'):
                url_complete_with_sas = ImageTIF800.query.get(numero_image)


            # url_complete_with_sas = url_du_blob +"images"+ dimension_img + "/" + image_name + ".tif" + token_sas
            print(url_complete_with_sas)
            blob_client = BlobClient.from_blob_url(url_complete_with_sas.chemin)
            blob_content = blob_client.download_blob().readall()

            encode_image = get_encoded_image(BytesIO(blob_content))


            coordinate,latitude,longitude = getCoordinate(url_complete_with_sas.chemin)

            response_data = {'image_paths': [{'path': numero_image, 'encoded_image': encode_image}],'gmap':{ 'link': coordinate,'latitude':latitude,'longitude':longitude, 'dimension': dimension_img,'url':url_complete_with_sas.chemin }}

            yield f"data: {json.dumps(response_data)}\n\n"


        # Signal that the processing is complete
    return Response(stream_with_context(generate_images(dimension_img)), content_type='text/event-stream')


@app.route('/update_image_start_suivant')
def update_image_start_suivant():
    # Mettre à jour la variable de session image_start
    session['image_start'] += 9
    return redirect(url_for('images'))
@app.route('/update_image_start_precedent')
def update_image_start_precedent():
    # Mettre à jour la variable de session image_start
    session['image_start'] -= 9
    return redirect(url_for('images'))

@app.route('/gallery.html')
def gallery():
    # Liste des chemins provenant de la base de données
    # chemins_images = CheminImageCouleur.query.filter_by(id_utilisateur=session['user']['id']).with_entities(CheminImageCouleur.chemin).all()
    chemins_images = CheminImageCouleur.query.filter_by(id_utilisateur=session['user']['id']).all()

    images_encodes = []
    for chem in chemins_images:
        # Chemin du fichier TIF
        chemin_tif = chem.chemin
        cheminSource_tif = chem.path_imgBase
        
        # Ouvrir le fichier TIF
        img_encode =get_encoded_image(chemin_tif)
        imgSource_encode = get_encoded_image(cheminSource_tif)

        images_encodes.append([img_encode,imgSource_encode,chem.name_model])

    return render_template('gallery.html', images_encodes=images_encodes)


def inference_and_insert(image_name, model,rand_value):
    # rand_value = random.randint(1, 999999999)
    inference_processus(model_name=model, image_path=image_name + ".tif", rand_value=rand_value)

    # with app.app_context():
    #     chemin_couleur = f"static/IA/output/pred/{rand_value}.tiff"
    #     image_couleur = CheminImageCouleur(chemin=chemin_couleur, id_utilisateur=session['user']['id'])

    #     db.session.add(image_couleur)
    #     db.session.commit()
@app.route('/inference.html',methods=['GET', 'POST'])
def inference():
    if 'connection' not in session:
        return redirect(url_for('connexion_compte'))

    dimension_img =  request.args.get('dimension_img', default='pas de num')
    model = request.args.get('model', default='valeur_par_defaut_autre')

    if model == 'valeur_par_defaut_autre' :
        return redirect(url_for('model'))

    extract_image =  request.args.get('image_extract')

    url_du_blob=""
    token_sas=""
    print(dimension_img )
    if extract_image:
        encode_image = get_encoded_image(extract_image)
        image_name = 'extract'
        local_path = f'/app/static/IA/data/sat/test/{image_name}.tif'

    else:

        if(dimension_img == '1600'):
            url_du_blob = "https://ia1.blob.core.windows.net/"
            token_sas = "?sp=racwdli&st=2023-12-13T15:43:50Z&se=2024-12-13T23:43:50Z&spr=https&sv=2022-11-02&sr=c&sig=JylybwxfntpEKAO6RcrTRPbekXL7otXDQS2DSs42kHk%3D"
        elif dimension_img == '800' : 
            url_du_blob = "https://ia2.blob.core.windows.net/"
            token_sas = "?sp=racwdli&st=2023-12-14T09:30:25Z&se=2024-12-14T17:30:25Z&spr=https&sv=2022-11-02&sr=c&sig=DvRdUrDozmLyhd1KbA8EtSwHHaC2Fu8Uv8sQWq10VCE%3D"
        numero_image =  request.args.get('numero_image', default='pas de num')
        if(dimension_img == '1600'):
            image_name = f"image_{dimension_img}_{numero_image}"
        elif(dimension_img == '800'):
            image_name = f"image_{numero_image}" 
        
        url_complete_with_sas = url_du_blob +"images"+ dimension_img + "/" + image_name + ".tif" + token_sas
        print("inf"+url_complete_with_sas)
        blob_client = BlobClient.from_blob_url(url_complete_with_sas)
        blob_content = blob_client.download_blob().readall()
        local_path = f'/app/static/IA/data/sat/test/{image_name}.tif'

        with open(local_path, "wb") as file:
            blob_data = blob_client.download_blob()
            file.write(blob_data.readall())
        encode_image = get_encoded_image(BytesIO(blob_content))
        values = request.args.get('values')
        # values = values.replace("'", "\"")
        print(model)


    result_queue = Queue()
    if request.method == 'POST':
        print(image_name+".tif")
        rand_value = random.randint(1, 999999999)
        def inference_and_insert(image_name, model, result_queue,rand_value):
            inference_processus(model_name=model, image_path=image_name + ".tif", rand_value=rand_value)



            # Mettre un résultat dans la file pour indiquer la fin du processus
            result_queue.put("Done")
        inference_thread = Thread(target=inference_and_insert, args=(image_name, model, result_queue,rand_value))
        inference_thread.start()

        # Attendre que le thread soit terminé
        inference_thread.join()
        with app.app_context():
            chemin_couleur = f"/app/static/IA/output/pred/{rand_value}.tiff"
            image_couleur = CheminImageCouleur(chemin=chemin_couleur, id_utilisateur=session['user']['id'],name_model=model,path_imgBase=local_path)

            db.session.add(image_couleur)
            db.session.commit()

        # Attendre que le processus ait fini avant de continuer
        return redirect(url_for('gallery'))  



    # Créez un flux mémoire pour stocker l'image PNG
    img_io = io.BytesIO()
    img_io.seek(0)

    image = request.args.get('image')
    # image = get_encoded_image(image) 

    if model == "classic_model.h5":
        description = "This Sequential Convolutional Neural Network (CNN) model is designed for image classification. It processes images of 25x25 pixels with 4 channels. The model consists of three convolutional layers, each followed by a MaxPooling2D layer, with increasing filter sizes (32, 64, 128) and 'relu' activation. After convolution, the data is flattened and passes through a dense layer with 128 neurons (also 'relu' activated), followed by a softmax output layer with 27 neurons, suitable for multi-class classification."
    elif model == "successive_model.h5":
        description = "This Convolutional Neural Network (CNN) model, named Successive Model, is designed for hierarchical and dependent image classification. It processes images of size 25x25 pixels with 4 channels. The model features three output layers, each predicting different class categories with a dependency chain: the prediction of the first layer (4 classes) influences the second (11 classes), and the second affects the third (27 classes). Compiled with the 'adam' optimizer and using 'categorical_crossentropy' for loss, it's ideal for complex, layered classification tasks."
    else:
        description = CheminModeleIA.query.filter_by(model_name=model.replace(".h5", "")).first().description


    model = model.replace(".h5", "")


    return render_template('inference.html', active_page='inference.html', model=model, image=image,encode_image=encode_image,description=description)

@app.route('/creation_compte.html',methods=['GET', 'POST'])
def creation_compte():
    if request.method == 'POST':
        # Récupérez les valeurs passées en tant que paramètres
        nom = request.form.get('nom')
        prenom = request.form.get('prenom')
        email = request.form.get('email')
        mot_de_passe = generate_password_hash(request.form.get('password'))

        user = Utilisateur.query.filter_by(email=email).first()

        # Vérifie que les champs ne sont pas vides
        if not (nom and prenom and email and mot_de_passe):
            flash("Tous les champs doivent être remplis", "error")
            return render_template('creation_compte.html')

        # Vérifie si l'email existe déjà
        user = Utilisateur.query.filter_by(email=email).first()
        if user:
            flash("Un compte avec cet email existe déjà", "error")
            return redirect(url_for('connexion_compte'))


        # Crée le compte utilisateur
        nouveau_utilisateur = Utilisateur(nom=nom, prenom=prenom, email=email, mot_de_passe=mot_de_passe)
        
        # session['user'] = nouveau_utilisateur

        db.session.add(nouveau_utilisateur)
        db.session.commit()
        flash("Compte créé avec succès!", "success")
        return redirect(url_for('home'))

    return render_template('creation_compte.html')

@app.route('/connexion_compte.html',methods=['GET', 'POST'])
def connexion_compte():
    if request.method == 'POST':
        # Récupérez les valeurs passées en tant que paramètres
        email = request.form.get('email')
        mot_de_passe = request.form.get('password')

        user = Utilisateur.query.filter_by(email=email).first()

        # Vérifie que les champs ne sont pas vides
        if not (email and mot_de_passe):
            flash("Tous les champs doivent être remplis", "error")
            return redirect(url_for('connexion_compte'))

        # Vérifie si l'email existe
        user = Utilisateur.query.filter_by(email=email).first()
        if not (user):
            flash("Aucun compte n'est associé à cette email", "error")
            return redirect(url_for('creation_compte'))

        # Récupère le compte utilisateur
        if not (check_password_hash(user.mot_de_passe, mot_de_passe)):
            flash("Mot de passe incorrect", "error")
            return redirect(url_for('connexion_compte'))
        
        user_info = {
            'id': user.id,
            'nom': user.nom,
            'prenom': user.prenom,
            'email': user.email,
            'mdp': user.mot_de_passe

        }
        session['user'] = user_info
        session['connection'] = 'success'
        flash("Connecté avec succès!", "success")
        return redirect(url_for('home'))

    return render_template('connexion_compte.html')

@app.route('/deconnexion_compte.html')
def deconnexion_compte():
        session.pop('connection')
        session.pop('user')

        flash("Vous avez été déconnecté avec succès!", "error")
        return redirect(url_for('home'))

@app.route('/info_compte.html')
def info_compte():
    if 'connection' not in session:
        return redirect(url_for('connexion_compte'))
    return render_template('info_compte.html')

@app.route('/progress', methods=["GET"])
def get_progress():
    return jsonify(progress=get_global_variable_progress())

@app.route('/progress_model_creation', methods=["GET"])
def get_progress_model_creation():
    progress, spin_active = get_global_variable_progress_model_creation()
    return jsonify(progress=progress, spin_active=spin_active)

def get_encoded_image(image_path):
    # Chargez l'image TIFF
    tif_image = Image.open(image_path)

    # Vérifiez si l'image a un canal alpha
    if "A" in tif_image.getbands():
        # Convertissez en utilisant le canal alpha
        png_image = tif_image.convert("RGBA")
    else:
        # Convertissez sans canal alpha
        png_image = tif_image.convert("RGB")

    # Créez un flux mémoire pour stocker l'image PNG
    img_io = io.BytesIO()
    png_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Encodez l'image en base64
    encoded_image = base64.b64encode(img_io.read()).decode('utf-8')
    return encoded_image
# ...

@app.route('/convert', methods=['POST'])
def convert_to_tiff():
    try:
        data = request.json
        img_encode = data.get('imgEncode')

        # Convertir la base64 en tableau de bytes
        byte_data = base64.b64decode(img_encode)

        # Créer une image à partir des bytes
        img = Image.open(io.BytesIO(byte_data))

        # Sauvegarder l'image en tant que fichier TIFF
        tiff_data = io.BytesIO()
        img.save(tiff_data, format='TIFF')

        # Obtenez le chemin du répertoire statique
        static_dir = os.path.join(os.path.dirname(__file__), 'static')

        # Enregistrez le fichier TIFF dans le répertoire statique
        tiff_filename = 'image.tiff'
        tiff_path = os.path.join(static_dir, tiff_filename)
        with open(tiff_path, 'wb') as tiff_file:
            tiff_file.write(tiff_data.getvalue())

        # Renvoyer l'URL du fichier TIFF au client
        return jsonify({'tiffUrl': f'/static/{tiff_filename}'})
    except Exception as e:
        print(f"Error during conversion and download: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

with app.app_context():
    db.create_all()
    image = ImageTIF800.query.filter_by().first()
    if not (image):
        with open('./test_db/scriptSQLReunion.sql', 'r') as file:
            print("Enregistrement des données images 800")
            for i in tqdm(range(1, 4403)):
                chemin = f"https://ia2.blob.core.windows.net/images800/image_{i}.tif?sp=racwdli&st=2023-12-14T09:30:25Z&se=2024-12-14T17:30:25Z&spr=https&sv=2022-11-02&sr=c&sig=DvRdUrDozmLyhd1KbA8EtSwHHaC2Fu8Uv8sQWq10VCE%3D"
                nouvelle_image = ImageTIF800(chemin=chemin)
                #google_maps_link,latitude,longitude = getCoordinate(chemin)
                #nouvelle_image = ImageTIF800(chemin=chemin,google_maps_link=google_maps_link,latitude=latitude,longitude=longitude)
                db.session.add(nouvelle_image)
            print("Enregistrement des données images 1600")
            for i in tqdm(range(1, 1147)):
                chemin = f"https://ia1.blob.core.windows.net/images1600/image_1600_{i}.tif?sp=racwdli&st=2023-12-13T15:43:50Z&se=2024-12-13T23:43:50Z&spr=https&sv=2022-11-02&sr=c&sig=JylybwxfntpEKAO6RcrTRPbekXL7otXDQS2DSs42kHk%3D"
                nouvelle_image = ImageTIF1600(chemin=chemin)
                #google_maps_link,latitude,longitude = getCoordinate(chemin)
                #nouvelle_image = ImageTIF1600(chemin=chemin,google_maps_link=google_maps_link,latitude=latitude,longitude=longitude)
                db.session.add(nouvelle_image)
            db.session.commit()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
