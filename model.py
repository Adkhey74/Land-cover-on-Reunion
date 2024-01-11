# model.py
from extensions import db

class Utilisateur(db.Model):
    __tablename__ = 'utilisateurs'

    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(100))
    prenom = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True, nullable=False)
    mot_de_passe = db.Column(db.String(500), nullable=False)

    def __repr__(self):
        return f'<Utilisateur {self.nom}>'

    
class CheminModeleIA(db.Model):
    __tablename__ = 'chemins_modeles_ia'

    id = db.Column(db.Integer, primary_key=True)
    id_utilisateur = db.Column(db.Integer, db.ForeignKey('utilisateurs.id'), nullable=False)
    chemin = db.Column(db.String(255), nullable=False)
    description = db.Column(db.String(255), nullable=False)

    image_height = db.Column(db.Integer, default=25)
    image_width = db.Column(db.Integer, default=25)
    pixel_resolution = db.Column(db.Integer, default=2)
    f_height = db.Column(db.Integer, default=0)
    f_width = db.Column(db.Integer, default=0)
    tile_ratio = db.Column(db.Float, default=0.8)
    nombre_couche = db.Column(db.Integer, default=3)
    nombre_neurone = db.Column(db.PickleType, default=[32, 64, 128])  # Liste stockée sous forme sérialisée
    filter_size = db.Column(db.Integer, default=3)
    pool_size = db.Column(db.Integer, default=2)
    activation_function = db.Column(db.String(50), default='relu')
    optimizer = db.Column(db.String(50), default='adam')
    loss = db.Column(db.String(50), default='categorical_crossentropy')
    metrics = db.Column(db.String(50), default='accuracy')
    patience_early_stopping = db.Column(db.Integer, default=5)
    restore_best_weights = db.Column(db.Boolean, default=True)
    min_delta_early_stopping = db.Column(db.Float, default=0.0001)
    mode = db.Column(db.String(50), default='max')
    factor = db.Column(db.Float, default=0.1)
    patience_reduce_lr = db.Column(db.Integer, default=5)
    min_lr_reduce_lr = db.Column(db.Float, default=0.00001)
    min_delta = db.Column(db.Float, default=0.0001)
    epochs = db.Column(db.Integer, default=100)
    model_name = db.Column(db.String(255), default="model_base")

    # Relation avec Utilisateur
    utilisateur = db.relationship('Utilisateur', backref=db.backref('modeles_ia', lazy=True))

    def __repr__(self):
        return f'<CheminModeleIA {self.chemin}>'
    
class CheminImageCouleur(db.Model):
    __tablename__ = 'chemins_images_couleur'

    id = db.Column(db.Integer, primary_key=True)
    id_utilisateur = db.Column(db.Integer, db.ForeignKey('utilisateurs.id'), nullable=False)
    name_model = db.Column(db.String(255), nullable=False)
    path_imgBase = db.Column(db.String(255), nullable=False)
    chemin = db.Column(db.String(255), nullable=False)

    # Relation avec 
    utilisateur = db.relationship('Utilisateur', backref=db.backref('images_couleur', lazy=True))

    def __repr__(self):
        return f'<CheminImageCouleur {self.chemin}>'

    
class ImageTIF800(db.Model):
    __tablename__ = 'images_tif_800'

    id = db.Column(db.Integer, primary_key=True)
    chemin = db.Column(db.String(500), nullable=False)
    # latitude = db.Column(db.String(255), nullable=False)
    # longitude = db.Column(db.String(255), nullable=False)
    # google_maps_link = db.Column(db.String(500), nullable=False)

    def __repr__(self):
        return f'<ImageTIF {self.chemin}>'


class ImageTIF1600(db.Model):
    __tablename__ = 'images_tif_1600'

    id = db.Column(db.Integer, primary_key=True)
    chemin = db.Column(db.String(500), nullable=False)
    # latitude = db.Column(db.String(255), nullable=False)
    # longitude = db.Column(db.String(255), nullable=False)
    # google_maps_link = db.Column(db.String(500), nullable=False)

    def __repr__(self):
        return f'<ImageTIF {self.chemin}>'

