# Utiliser une image Python de base plus complète
FROM python:3.8

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système requises pour certaines bibliothèques
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

ENV DOCKERIZE_VERSION v0.6.1
RUN wget https://github.com/jwilder/dockerize/releases/download/$DOCKERIZE_VERSION/dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && tar -C /usr/local/bin -xzvf dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
    && rm dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz


# Définir les variables d'environnement pour GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Installer les dépendances Python
RUN pip install Flask numpy pandas tqdm scikit-learn tensorflow keras
RUN pip install GDAL==$(gdal-config --version) # Assurez-vous que la version de GDAL correspond
RUN pip install geopandas
RUN pip install Pillow
RUN pip install flask-sqlalchemy
RUN pip install psycopg2
RUN pip install werkzeug
RUN pip install azure-storage-blob
RUN pip install flask-cors
# Exposer le port sur lequel l'application Flask s'exécute
EXPOSE 5000

