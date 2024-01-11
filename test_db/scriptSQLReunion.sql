-- Création de la table des utilisateurs
CREATE TABLE utilisateurs (
    id SERIAL PRIMARY KEY,
    nom VARCHAR(100),
    email VARCHAR(100) UNIQUE NOT NULL,
    mot_de_passe VARCHAR(100) NOT NULL
);

-- Création de la table des chemins de modèles d'IA
CREATE TABLE chemins_modeles_ia (
    id SERIAL PRIMARY KEY,
    id_utilisateur INT REFERENCES utilisateurs(id),
    chemin VARCHAR(255) NOT NULL
);

-- Création de la table des images TIF
CREATE TABLE images_tif (
    id SERIAL PRIMARY KEY,
    chemin VARCHAR(255) NOT NULL
);


DO $$
DECLARE
    i INT := 1;
BEGIN
    WHILE i <= 4402 LOOP
        INSERT INTO images_tif (chemin)
        VALUES ('https://landcoverreuniontif.blob.core.windows.net/imagestif/image_' || i || '.tif' || '?sp=r&st=2023-12-12T09:59:18Z&se=2023-12-12T17:59:18Z&spr=https&sv=2022-11-02&sr=c&sig=r%2FRBpWAzlUhJFaE%2BxYwngZ21dPb5Ka2U1T38HD%2Fhi7o%3D');
        i := i + 1;
    END LOOP;
END $$;
