from osgeo import gdal
from pyproj import Proj, Transformer

def getCoordinate(link):
    # Ouvrir le fichier TIFF
    # tiff_file = 'static/IA/data/sat/test/extract.tif'
    tiff_file = link
    dataset = gdal.Open(tiff_file)

    # Récupérer les informations de géoréférencement
    geotransform = dataset.GetGeoTransform()

    # Définir les projections
    utm_zone_40S = Proj(proj="utm", zone=40, south=True, ellps='GRS80', units='m', no_defs=True)
    wgs84 = Proj(proj='latlong', datum='WGS84')

    # Coordonnées du coin supérieur gauche de votre image
    x = geotransform[0]
    y = geotransform[3]

    # Convertir les coordonnées UTM en coordonnées de latitude et longitude
    transformer = Transformer.from_proj(utm_zone_40S, wgs84)
    longitude, latitude = transformer.transform(x, y)

    # Maintenant, latitude et longitude contiennent les coordonnées dans le format utilisé par Google Maps
    print(f"Latitude: {latitude}, Longitude: {longitude}")

    def generate_google_maps_link(lat, long):
        base_url = "https://www.google.com/maps/search/?api=1&query="
        return f"{base_url}{lat},{long}"


    google_maps_link = generate_google_maps_link(latitude, longitude)
    return google_maps_link,latitude,longitude