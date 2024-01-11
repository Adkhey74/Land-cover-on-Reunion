#Import required libraries
import numpy as np
import pandas as pd
import time                                     #Check code calculation time
from tqdm import tqdm
from osgeo import gdal, ogr     #Open and manipulate geo tiff files
import geopandas as gpd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense


from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



def model_creation(image_height = 25, image_width = 25, pixel_resolution = 2, f_height = 0, f_width = 0, tile_ratio = 0.8, nombre_couche = 3, nombre_neurone = [32, 64, 128], filter_size = 3, pool_size = 2, activation_function = "relu", optimizer = "adam", loss = "categorical_crossentropy", metrics = "accuracy", patience_early_stopping = 5, restore_best_weights = True, min_delta_early_stopping = 0.0001, mode = 'max', factor = 0.1, patience_reduce_lr = 5, min_lr_reduce_lr = 0.00001, min_delta = 0.0001, epochs = 100, model_name = "model_base"):
    extract_tiles(image_height, image_width, pixel_resolution, f_height, f_width, tile_ratio)
    dataX, dataY, train_label, test_label, val_label, idTrain, idTest, idVal = generate_data()
    model = create_model(image_height, image_width, nombre_couche, nombre_neurone, filter_size, pool_size, activation_function, optimizer, loss, metrics)
    datagen_X, images_test = model_training(dataX, dataY, train_label, test_label, val_label, idTrain, idTest, idVal, model, patience_early_stopping, restore_best_weights, min_delta_early_stopping, mode, factor, patience_reduce_lr, min_lr_reduce_lr, min_delta, epochs, model_name)
    test_model(datagen_X, images_test, test_label, model)

total_operations = 0
current_operation = 0
progress_percentage = 0
spin_active = False
def extract_tiles(image_height, image_width, pixel_resolution, f_height, f_width, tile_ratio):
    #Define input and output
    input_path   = '/app/static/IA/data/'
    output_path  = '/app/static/IA/output/set/'
    mp           = 'sat/2020-2m.tif'
    sh           = 'true/2020.shp'
    pixelRes     = pixel_resolution
    rasterFormat = 'GTiff'
    vectorFormat = 'ESRI Shapefile'
    print('Input and output loaded')


    #Définition de la taille imagette
    img_height,img_width = image_height,image_width
    #Calcul de l'aire imagette
    maxArea = (img_width*pixelRes)*(img_width*pixelRes)
    #Definition du critere de selection
    tileRatio = tile_ratio
    #Definition d'une cadre, par défaut 0 pour cadre égale à la taille imagette pré-définie
    frame_height, frame_width = f_height,f_width
    print('[3/10] Parameters defined')


    def extractTile(Rasters,Shapes,idY,output_path):

        rasterArrays = []
        lead = []

        global total_operations
        global current_operation
        global progress_percentage

        current_operation = 0
        for i in range(len(Shapes)):
            rt = Rasters[i]
            sh = Shapes[i]
            #Load raster one by one
            raster = gdal.Open(rt)
            #Load shape one by one
            ds = gpd.read_file(sh)
            #Nombre de polygones
            nEntity,details = ds.shape
            #Recuperer emprise du raster complet
            gt = raster.GetGeoTransform()

            for entity in tqdm(range(0,nEntity-1)):
                #Informations sur le polygone cible
                infosParcelle = ds.iloc[entity]
                #Géométrie point par point du polygone cible
                polyParcelle = str(ds.iloc[entity].geometry)
                #Calcul de l'enveloppe du polygone cible
                geom = ogr.CreateGeometryFromWkt(polyParcelle)
                env = geom.GetEnvelope()

                #Calcul de la largeur et longueur projetée de la parcelle
                width = (env[1]-env[0])/pixelRes
                height = (env[3]-env[2])/pixelRes
                #Calcul du nombre de tuiles possibles
                Rxtile = width/img_width
                Rytile = height/img_height
                Xtile = int(Rxtile)+1
                Ytile = int(Rytile)+1

                total_operations = nEntity

                current_operation = entity + 1
                progress_percentage = (current_operation / total_operations) * 100

                #Boucle pour la génération des imagettes avec condition
                xCount = 0
                yCount = 0
                for yy in range(Ytile):
                    yMin = env[2]+((img_height*pixelRes)*(yy))
                    yMax = env[2]+((img_height*pixelRes)*(yy+1))
                    for xx in range(Xtile):
                        xMin = env[0]+((img_width*pixelRes)*(xx))
                        xMax = env[0]+((img_width*pixelRes)*(xx+1))
                        #Create the square polygon the Ring
                        ring = ogr.Geometry(ogr.wkbLinearRing)
                        ring.AddPoint(xMin,yMin) #Start ring
                        ring.AddPoint(xMax,yMin)
                        ring.AddPoint(xMax,yMax)
                        ring.AddPoint(xMin,yMax)
                        ring.AddPoint(xMin,yMin) #Close ring
                        # Create polygon
                        poly = ogr.Geometry(ogr.wkbPolygon)
                        poly.AddGeometry(ring)
                        poly.ExportToWkt()
                        #Conditional generation by intersection
                        poly1 = ogr.CreateGeometryFromWkt(polyParcelle)
                        poly2 = ogr.CreateGeometryFromWkt(str(poly))
                        intersection = poly1.Intersection(poly2)
                        interArea = intersection.GetArea()
                        if interArea >= maxArea*tileRatio:     #Condition de recouvrements
                            #Convert raster tile to array
                            xo = int(round((xMin-gt[0])/pixelRes))
                            yo = int(round((gt[3]-yMax)/pixelRes))
                            arr = raster.ReadAsArray(xo-frame_width,yo-frame_height,
                                                    img_width+(frame_width+frame_width),
                                                    img_height+(frame_height+frame_height))
                            rasterArrays.append(arr.transpose(1, 2, 0))
                            #Create target
                            lead.append([idY,entity,infosParcelle.Code1,infosParcelle.Code2,infosParcelle.Code3])
                            #Generate tile tiff
            #              tile = gdal.Warp(output_path+str(len(lead)-1)+'.tiff', raster, format=rasterFormat, outputType=gdal.gdalconst.GDT_Byte,
            #                              outputBounds=[xMin, yMin, xMax, yMax],
                #                             xRes=pixelRes, yRes=pixelRes,
                #                            dstSRS='EPSG:32740', resampleAlg=None, options=['COMPRESS=NONE'])
                        xCount += 1
                    yCount += 1
                    xCount = 0
            raster = None
            ds = None

        #Save X
        X = np.asarray(rasterArrays, dtype='uint32')
        print("Input shape: "+str(X.shape))
        np.save(output_path+'X_'+str(idY), X)
        print('Saved numpy input')

        #Save Y
        Y = np.asarray(lead, dtype='uint32')
        print("Target shape: "+str(Y.shape))
        np.save(output_path+'Y_'+str(idY), Y)
        print('Saved numpy target')

    extractTile([input_path+mp],[input_path+sh],2020,output_path)

    global spin_active
    spin_active = True


def generate_data():
    dataX = np.load('/app/static/IA/output/set/X_2020.npy')
    dataY = np.load('/app/static/IA/output/set/Y_2020.npy')


    df = pd.DataFrame(dataY)

    # Obtenez une liste unique des IDs
    unique_ids = df[1].unique()

    # Divisez les IDs en train, test, et validation
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.4, random_state=42)  # 60% pour le training
    test_ids, val_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)  # 20% pour le test et 20% pour la validation

    # Utilisez ces IDs pour diviser le DataFrame
    train_label = df[df[1].isin(train_ids)]
    test_label = df[df[1].isin(test_ids)]
    val_label = df[df[1].isin(val_ids)]

    # print(train_label.shape)
    # print(train_label.shape[0]*100/23977)
    # print(test_label.shape)
    # print(test_label.shape[0]*100/23977)
    # print(val_label.shape)
    # print(val_label.shape[0]*100/23977)

    # print("Train data:\n", train_label)
    # print("\nTest data:\n", test_label)
    # print("\nValidation data:\n", val_label)

    idTrain = np.unique(train_label[1])
    idTest = np.unique(test_label[1])
    idVal = np.unique(val_label[1])

    return dataX, dataY, train_label, test_label, val_label, idTrain, idTest, idVal 



def create_model(image_height, image_width, nombre_couche, nombre_neurone, filter_size, pool_size, activation_function, optimizer, loss, metrics):
    model = Sequential()

    for i in range(nombre_couche) :
        if i == 0 :
            # Première couche de convolution
            model.add(Conv2D(nombre_neurone[i], (filter_size, filter_size), activation=activation_function, input_shape=(image_height, image_width, 4), padding='same'))
            model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
        else :
            model.add(Conv2D(nombre_neurone[i], (filter_size, filter_size), activation=activation_function, padding='same'))
            model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    # Flattening
    model.add(Flatten())

    # Couche dense
    model.add(Dense(nombre_neurone[nombre_couche-1], activation=activation_function))
    model.add(Dense(27, activation='softmax'))  # Utilisez 'softmax' si vous avez plus de 2 classes et ajustez le nombre de neurones en conséquence


    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def model_training(dataX, dataY, train_label, test_label, val_label, idTrain, idTest, idVal, model, patience_early_stopping, restore_best_weights, min_delta_early_stopping, mode, factor, patience_reduce_lr, min_lr_reduce_lr, min_delta, epochs, model_name):   
    datagen_X = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    indices_train = train_label.index.tolist()
    indices_test = test_label.index.tolist()
    indices_val = val_label.index.tolist()

    images_train = dataX[indices_train]
    images_test = dataX[indices_test]
    images_val = dataX[indices_val]


    datagen_X.fit(images_train)
    datagen_X.fit(images_val)

    image_generator_train = datagen_X.flow(images_train, encodingData(train_label), batch_size=10, shuffle=True)
    image_generator_val = datagen_X.flow(images_val, encodingData(val_label), batch_size=10, shuffle=True)

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience_early_stopping, restore_best_weights=restore_best_weights, min_delta=min_delta_early_stopping, verbose=1, mode=mode)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=factor, patience=patience_reduce_lr, min_lr=min_lr_reduce_lr, verbose=1, min_delta=min_delta)

    history = model.fit(image_generator_train, epochs=epochs, steps_per_epoch=len(train_label) // 10, validation_data=image_generator_val, validation_steps=len(val_label) // 10, callbacks=[early_stopping, reduce_lr])

    chemin_sauvegarde  = '/app/static/IA/model/' + model_name + '.h5'

    model.save(chemin_sauvegarde)
    print(f"Modèle sauvegardé avec succès à {chemin_sauvegarde}")

    return datagen_X, images_test


def test_model(datagen_X, images_test, test_label, model):
    image_generator_test = datagen_X.flow(images_test, shuffle=False)

    # Faire des prédictions sur les données de test
    predictions = model.predict(image_generator_test)

    # Si vous voulez convertir les prédictions en classes
    predicted_classes = np.argmax(predictions, axis=1)

    # Obtenir les étiquettes réelles
    real_labels = encodingData(test_label)

    real_classes = np.argmax(real_labels, axis=1)


    # Définissez les options d'affichage pour afficher le tableau complet
    np.set_printoptions(threshold=np.inf)

    np.set_printoptions(threshold=1000)


    # Générer le rapport de classification
    print(classification_report(real_classes, predicted_classes))





    conf_matrix = confusion_matrix(real_classes, predicted_classes)

    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    support = np.sum(conf_matrix, axis=1)



# Fonction général

def encodingData(data_label):
        # Votre tableau d'origine
        classes_tier3 = np.array(data_label[[4]])

        # Déterminer la taille du tableau résultant
        n = 27  # Nombre total d'éléments dans chaque sous-tableau
        m = classes_tier3.shape[0]  # Nombre de sous-tableaux dans x

        # Créer un nouveau tableau rempli de zéros
        result = np.zeros((m, n), dtype=int)

        # Remplacer les éléments appropriés par ceux de x
        for i in range(m):
            result[i, classes_tier3[i]-1] = 1

        return result


def get_global_variable_progress_model_creation():
    return progress_percentage, spin_active