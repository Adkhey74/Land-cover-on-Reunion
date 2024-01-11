from azure.storage.blob import BlobClient

# URL du Blob
def get_image(image_name):
    url_du_blob = "https://ia1.blob.core.windows.net/images1600/"+image_name

    # Jeton SAS
    token_sas = "?sp=racwdli&st=2023-12-13T15:43:50Z&se=2024-12-13T23:43:50Z&spr=https&sv=2022-11-02&sr=c&sig=JylybwxfntpEKAO6RcrTRPbekXL7otXDQS2DSs42kHk%3D"

    # Combinez l'URL et le jeton SAS
    url_complete = url_du_blob + token_sas

    # Créez une instance de BlobClient en utilisant l'URL complète
    blob_client = BlobClient.from_blob_url(url_complete)

    # Téléchargez le fichier
    # with open("/app/images/image_2.tif", "wb") as download_file:
    #     download_file.write(blob_client.download_blob().readall())

    print("Téléchargement terminé.")
