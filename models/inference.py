#Import libraries
from osgeo import gdal
from tensorflow import keras
import gc
import numpy as np
from tqdm import tqdm
import time
import math
import glob, os
import re
import random
from flask import session
print('Libraries imported')


def inference_processus(image_height = 25, image_width = 25, padding_shift = 1, pixel_resolution = 2, model_name = "bestModel.h5",image_path = 'image_1600_1.tif',rand_value=random.randint(1, 999999999)):
    Xout, Yout, raster_path, datagen_X, gt, img_width, img_height, paddingShift, pixelRes, start, output_path, model, image_path = initialisation(image_height, image_width, padding_shift, pixel_resolution, model_name,image_path)

    inference_loop(Xout, Yout, raster_path, datagen_X, gt, img_width, img_height, paddingShift, pixelRes, start, output_path, model, model_name)

    inference_load(Xout, img_width, img_height, pixelRes,image_path,rand_value)





def initialisation(image_height, image_width, padding_shift, pixel_resolution, model_name,image_path):
    #Define input
    model_path  = '/app/static/IA/model/'
    raster_path = "/app/static/IA/data/sat/test/"+image_path
    output_path = "/app/static/IA/output/pred/"
    #Define tile size
    img_width,img_height = image_width,image_height
    #Define padding shift
    paddingShift = padding_shift
    #Define pixel resolution
    pixelRes     = pixel_resolution
    rasterFormat = 'GTiff'
    start = 0 #Change value if there is an inference-loop interuption case
    print('Input defined')


    #Load model
    model = keras.models.load_model(model_path+model_name)
    datagen_X = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    print('Inference model and dataGenerator loaded')


    #Load raster
    raster = gdal.Open(raster_path)
    #Recuperer emprise du raster complet
    gt = raster.GetGeoTransform()
    origin = [gt[0],gt[3]]
    raster_xSize = raster.GetRasterBand(1).XSize
    raster_ySize = raster.GetRasterBand(1).YSize
    print('Raster informations loaded')
    print("Xsize = %d and Ysize = %d" %(raster_xSize, raster_ySize))


    #Compute output size following raster size
    def human_format(num):
        num = float('{:.3g}'.format(num))
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

    if isinstance(img_height/2,int):
        #Case where tile size is odd - impair
        print("Tile size is odd")
    else:
        #Case where tile size is even - pair
        print("Tile size is even")

    Xout = math.ceil( ((raster_xSize-img_width)+1)/paddingShift )
    Yout = math.ceil( ((raster_ySize-img_height)+1)/paddingShift )
    matrixUnit = Xout * Yout
    print("Xin  = %d and Yin  = %d" %(raster_xSize, raster_ySize))
    print("Xout = %d and Yout = %d" %(Xout, Yout))
    print("Number of predictions:",human_format(matrixUnit))

    return Xout, Yout, raster_path, datagen_X, gt, img_width, img_height, paddingShift, pixelRes, start, output_path, model,image_path


progress = 0
def inference_loop(Xout, Yout, raster_path, datagen_X, gt, img_width, img_height, paddingShift, pixelRes, start, output_path, model, model_name):
    #Start inference loop
    print('[5/10] Start inference loop')
    rasterArrays   = []
    outputAllPred  = []
    outputSavePred = []
    concat = 100
    saving = 100

    print('Start inference at '+time.strftime("%Y-%m-%d %H:%M:%S"))
    start_Totaltime = time.time()
    global progress
    for yy in tqdm(range(start,Yout)):
        progress = (yy / Yout) * 100  # Mise à jour de la progression
        start_time = time.time()
        #Dimensions in meter
        yMin = gt[3]-((paddingShift*pixelRes)*(yy))-(img_height*pixelRes)
        yMax = gt[3]-((paddingShift*pixelRes)*(yy))
        raster = gdal.Open(raster_path)
        for xx in range(Xout):
            xMin = gt[0]+((paddingShift*pixelRes)*(xx))
            xMax = gt[0]+((paddingShift*pixelRes)*(xx))+(img_height*pixelRes)
            #Convert raster tile to array, meter to pixel number
            xo = int(round((xMin-gt[0])/pixelRes))
            yo = int(round((gt[3]-yMax)/pixelRes))
            arr = raster.ReadAsArray(xo,yo,img_width,img_height)
            rasterArrays.append(arr.transpose(1, 2, 0))

            #Generate tile tiff
        # tile = gdal.Warp(output_path+str(yy)+"_"+str(xx)+'.tiff', raster, format=rasterFormat, outputType=gdal.gdalconst.GDT_Byte,
            #                 outputBounds=[xMin, yMin, xMax, yMax],
            #                xRes=pixelRes, yRes=pixelRes,
            #               dstSRS='EPSG:32740', resampleAlg=None, options=['COMPRESS=NONE'])

        #Input and normalisation data
        if yy == start:
            #Initialize data generator
            X = np.array(rasterArrays).astype('float16')
            datagen_X.fit(X)

        yy_concat = yy + 1
        if ((yy_concat%concat==0) and (yy!=start)) | (yy==Yout-1):
            #Input and normalisation data
            print("Raster Arrays", len(rasterArrays))
            X = np.array(rasterArrays).astype('float16')
            print("X", X.shape)
            #print("X shape : ",X.shape)
            #Predict data with imported model

            if model_name == "successive_model.h5" :
                # Les mappings entre les tiers
                tier1_to_tier2 = {
                    0: np.arange(0, 5),
                    1: np.arange(5, 9),
                    2: np.arange(9, 10),
                    3: np.arange(10, 11),
                }

                tier2_to_tier3 = {
                    0: np.arange(0, 1),
                    1: np.arange(1, 3),
                    2: np.arange(3, 5),
                    3: np.arange(5, 6),
                    4: np.arange(6, 11),
                    5: np.arange(11, 14),
                    6: np.arange(14, 20),
                    7: np.arange(20, 21),
                    8: np.arange(21, 22),
                    9: np.arange(22, 24),
                    10: np.arange(24, 27),
                }

                outputPred = model.predict(X)

                Yhat_format = []
                for i in range(len(outputPred[0])):
                    # Prédiction de la classe de tier 1
                    tier1_pred = np.argmax(outputPred[0][i])
                    # Sélection des indices de tier 2 basés sur la classe prédite de tier 1
                    tier2_indices = tier1_to_tier2[tier1_pred]
                    # Prédiction de la classe de tier 2
                    tier2_pred = np.argmax(outputPred[1][i][tier2_indices]) + tier2_indices[0]

                    # Sélection des indices de tier 3 basés sur la classe prédite de tier 2
                    tier3_indices = tier2_to_tier3[tier2_pred]
                    # Prédiction de la classe de tier 3
                    tier3_pred = np.argmax(outputPred[2][i][tier3_indices]) + tier3_indices[0]

                    #laPred = np.argmax(pred, axis=1)
                    Yhat_format.append(tier3_pred)
            else :
                outputPred = model.predict(datagen_X.flow(X,batch_size=32,shuffle=False),verbose=0)
                Yhat_format = np.argmax(outputPred, axis=1)  # ///////////////////////////////////////////////// Changement car pas le même modèle, nous 1 branche, lui 4
            
            keras.backend.clear_session()
            gc.collect()
            #Concatenate for txt save
            outputSavePred.append(Yhat_format)

            if (yy_concat%saving==0) and (yy!=start):
                #Write the Line Yhat_format
                np.save(output_path+str(int(yy_concat/100))+'.npy',np.vstack(outputSavePred))
                #Clear output
                outputSavePred.clear()

            #Clear tiles data
            rasterArrays.clear()
            del X

            end_time = time.time()
            hours, rem = divmod(end_time-start_time,3600)
            minutes, seconds = divmod(rem,60)

    #     print("Iteration "+str(yy)+", "+
        #          "time computation "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)+", "+
        #         "remaining "+str(int((Yout/1)-yy))+" iterations, "+
        #        "estimated time "+ str( ((end_time-start_time)/3600) *((Yout/concat)-(yy/concat)) )+" hours.")

    np.save(output_path+str(int(Yout/saving)+1)+'.npy',np.vstack(outputSavePred))

    end_time = time.time()
    days, remd = divmod(end_time-start_Totaltime,86400)
    hours, remh = divmod(remd,3600)
    minutes, seconds = divmod(remh,60)
    print("Total time "+"{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    print('[6/10] End inference at '+time.strftime("%Y-%m-%d %H:%M:%S"))


def inference_load(Xout, img_width, img_height, pixelRes,image_path,rand_value):
    raster_path = "/app/static/IA/data/sat/test/"+image_path
    output_path = "/app/static/IA/output/pred/"

    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    os.chdir(output_path)
    classif = []

    #Load numpy file
    for infile in sorted(glob.glob('*.np[yz]'), key=numericalSort):
        print("Current File Being Processed is: " + infile)
        brut = np.load(infile,allow_pickle=True).astype(int)

        # Pour la convertir en liste
        brut = brut + 1
        liste_brut = brut.flatten().tolist()
        classif.extend(liste_brut)



    # Transformation aux format de l'image

    matrices_reshape = []

    for i in range(0, len(classif), Xout):
        ligne = classif[i:i+Xout]

        if len(ligne) == Xout:
            matrices_reshape.append(np.array([ligne]))
        else:
            print("Erreur, mauvais nombre de données par ligne")


    data_plot = np.vstack(matrices_reshape)




    #data_plot = np.vstack(classif)

    #matrice_finale = data_plot.reshape((1545, 1499))

    del classif


    # from matplotlib.pyplot import imshow
    # imshow(data_plot)

    #Load raster to get emprise
    raster = gdal.Open(raster_path)
    #Recuperer emprise du raster complet
    gt     = raster.GetGeoTransform()
    sr     = raster.GetProjection()
    origin = [gt[0],gt[3]]

    #Make geotransform
    xmin,ymax = [origin[0]+(((img_width/2))*pixelRes),origin[1]-(((img_height/2))*pixelRes)]
    raster_size = np.shape(data_plot)
    nrows,ncols = raster_size
    xres = pixelRes
    yres = pixelRes
    geotransform = (xmin,xres,0,ymax,0, -yres)

    #Write output
    # driver = gdal.GetDriverByName('Gtiff')
    # dataset = driver.Create(output_path+".tiff", ncols, nrows, 1, gdal.GDT_Byte)
    # dataset.SetGeoTransform(geotransform)
    # dataset.SetProjection(sr)
    # dataset.GetRasterBand(1).WriteArray(data_plot)
    # dataset = None


    # Transformation de la matrice de classe (2D) en matrice RGB (3D)

    # Code RGB correspondant aux classes
    correspondance_rgb = [[251, 254, 42],[68, 244, 20],[83, 235, 184],[251, 116, 5],[215, 215, 158],[204, 1, 1],[112, 48, 160],[184, 137, 219],[220, 197, 237],[221, 79, 194],[137, 27, 116],[54, 77, 31],[114, 172, 62],[1, 168, 132],[169, 208, 142],[132, 191, 77],[202, 193, 12],[124, 118, 7],[131, 211, 26],[154, 223, 90],[182, 116, 18],[96, 96, 96],[0, 118, 142],[1, 204, 204],[255, 1, 1],[255, 171, 171],[210, 178, 197]]

    def classe_vers_rgb(classe):
        return correspondance_rgb[int(classe - 1)]

    matrice_rgb = np.array([[classe_vers_rgb(classe) for classe in ligne] for ligne in data_plot])

    print(matrice_rgb.shape)


    # Version RGB



    #Write output
    driver = gdal.GetDriverByName('Gtiff')
    # Créer un dataset TIFF avec 3 bandes pour RGB
    datasetRGB = driver.Create(output_path+f"{rand_value}.tiff", ncols, nrows, 3, gdal.GDT_Byte)
    datasetRGB.SetGeoTransform(geotransform)
    datasetRGB.SetProjection(sr)

    # Assurez-vous que data_plot est une matrice (1499, 1545, 3) avant d'exécuter ce code
    # Écrire les données de chaque canal dans les bandes correspondantes
    for i in range(3): # Pour R, G et B
        datasetRGB.GetRasterBand(i+1).WriteArray(matrice_rgb[:,:,i])

    # Fermeture propre du fichier pour enregistrer les données
    datasetRGB = None

    # Chemin du dossier dont vous voulez supprimer les fichiers
    folder_path = '/app/static/IA/output/pred'

    # Boucler sur tous les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Suppression du fichier
                os.unlink(file_path)
            except Exception as e:
                print('Échec de suppression de %s. Raison: %s' % (file_path, e))


def get_global_variable_progress():
    return progress