from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv
import os
import json

load_dotenv()
KEY_PATH = os.getenv("KEY_PATH")
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
storage_client = storage.Client(credentials=credentials)


def get_google_file(filename: str) -> bytes:
    bucket_name = "image-storage-21042024"
    blob_name = "images_compressed/" + filename
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    file_stream = blob.download_as_bytes()
    return file_stream
