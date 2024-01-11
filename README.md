# Land cover on Reunion

Land Cover on Reunion est une application innovante conçue pour cartographier l'île de la Réunion en identifiant de manière détaillée et précise tous ses terrains. Chaque type de terrain est distingué par une couleur spécifique, permettant une visualisation claire et intuitive de la diversité géographique de l'île.

Ce projet tire parti de modèles de réseaux de neurones convolutifs (CNN), développés avec TensorFlow, une plateforme d'apprentissage automatique de pointe. Ces modèles CNN sont spécialement conçus pour analyser et interpréter des images complexes, rendant notre application particulièrement efficace dans la reconnaissance et la classification des différents types de couvertures terrestres.

Destinée à un large éventail d'utilisateurs, de passionnés de géographie aux professionnels du domaine, Land Cover on Reunion se veut accessible et utile pour tous. Que ce soit pour des projets éducatifs, de recherche, ou simplement pour satisfaire la curiosité, notre application offre une perspective unique sur l'environnement naturel de l'île.

L'un des principaux avantages de Land Cover on Reunion est sa capacité à fournir des informations détaillées et précises sur le paysage de l'île, ce qui peut être bénéfique pour une variété d'applications, allant de la planification urbaine à la conservation de la nature. Cette approche innovante non seulement facilite la compréhension de la géographie complexe de la Réunion, mais ouvre également la voie à de nouvelles méthodes d'exploration et de préservation de l'environnement.

## Prérequis

L'un des avantages majeurs de Land Cover on Reunion est sa facilité d'installation et de configuration grâce à l'utilisation de Docker. Aucun prérequis spécifique n'est nécessaire en termes de bibliothèques ou d'environnements de programmation. Toutes les dépendances et configurations nécessaires sont gérées automatiquement par Docker. Cela rend l'application facilement accessible et opérationnelle pour tous les utilisateurs, indépendamment de leur système d'exploitation ou de leur configuration matérielle.

## Collaborateurs et Rôles

- **Samuel Pochat**: Développement, intégration des modèles d’IA dans l’application Flask et amélioration interfaces.
- **Dimitri Molina**: Développement, intégration des modèles d’IA dans l’application Flask et amélioration interfaces.
- **Adil Khadich**: Développement des interfaces utilisateur et intégration des programmes IA dans l’application Flask.
- **Adrien Marchetti** : Rien fait (démissionnaire depuis le début de l'année).

## À propos de l'Application

Cette application offre la possibilité de créer des modèles d'IA personnalisés ainsi que d'utiliser des modèles préconçus pour cartographier l'île de la Réunion. Les fonctionnalités clés incluent :

- Création et gestion de compte utilisateur.
- Possibilité de lier des modèles d'IA au compte de l'utilisateur.
- Présence de 2 modèles de base :
  - Un modèle CNN classique
  - Un modèle CNN multiclasses
- Sauvegarde des images générées.
- Utilisation d'une base de données PostgreSQL hébergée sur Docker.
- Ensemble d'images de test comprenant :
  - 4446 images de 800x800 mètres.
  - 1146 images de 1600x1600 mètres.
  - 1 images ultra haute résolution
- Stockage des images sur un serveur externe (Azure).

## Installation

1. Cloner le dépôt GitHub :
   ```
   git clone https://github.com/Samuelp74/Land-cover-on-Reunion.git
   ```
2. Télécharger l'image de base :
   ```
   https://drive.google.com/file/d/1IH2cpnHQQHqY6MQETIiA2uggZNQ8_I_t/view?usp=sharing
   ```
3. Copier l'image dans le dossier :
   ```
   /static/IA/data/sat
   ```
4. Construire et démarrer le service avec Docker :
   ```
   docker-compose up --build
   ```
5. Après le chargement complet du projet, accéder à l'application via :
   ```
   http://localhost:5000
   ```

## Utilisation

- **Création de Modèle 100% personnalisé** : Accéder à l'onglet ‘Dataset’ pour créer un modèle et suivre les étapes indiquées.
- **Utilisation de Modèle Existant** : Sélectionner un modèle via l'onglet ‘Model’. Les options incluent des modèles de base et des modèles liés au compte utilisateur.
- **Traitement d'Image** : Choisir une image de test dans l'onglet ‘Images’, puis lancer l'inférence en cliquant sur ‘Start Inference’ dans le menu d'inférence.
- **Observer le résultat de l'inférence** : Se rendre dans l'onget 'Gallery Picture'.

## Accès à la base de données

Pour accéder à la base de données :

1. Ouvrir pgAdmin
2. Clic droit sur Servers > Nouveau > Serveur
3. Définir un nom, exemple : 'Docker reunion'
4. Dans connection, 'nom d'hôte' : 'localhost', 'Port' : 'postgres', 'Mot de passe' : 'password'
5. Puis save, et se rendre sur 'mydatabase'
