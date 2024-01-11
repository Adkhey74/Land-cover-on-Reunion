import os

# Définir le chemin vers le répertoire avec les fichiers .tif
dir_path = 'D:/reunion/Land-cover-on-Reunion/static/images_tif'

# Lister tous les fichiers .tif dans le répertoire
tif_files = [f for f in os.listdir(dir_path) if f.endswith('.tif')]

# Trier la liste des fichiers pour s'assurer de l'ordre si nécessaire
tif_files.sort()

# Boucler sur la liste des fichiers et les renommer
for i, tif in enumerate(tif_files, start=1):
    # Définir le nouveau nom du fichier avec un numéro de séquence
    new_name = f"image_{i}.tif"
    old_file_path = os.path.join(dir_path, tif)
    new_file_path = os.path.join(dir_path, new_name)
    
    # Renommer le fichier
    os.rename(old_file_path, new_file_path)
    
    # Afficher le nom du fichier avant et après le renommage
    print(f"Renommé: {tif} en {new_name}")
