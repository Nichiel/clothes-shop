from PIL import Image
from fashion_clip.fashion_clip import FashionCLIP
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO

import src.configuration as configuration
from src.google_cli import get_google_file
from src.image_processor import ImageProcessing, ClipProcessing
from src.qdrant import query_collection
from src.utils import put_images_to_zip, check_is_type_available

app = FastAPI()
model_for_image = ImageProcessing(configuration.model_for_image_name)
model_for_text = ClipProcessing(configuration.model_for_text_name)


@app.get("/")
async def root():
    return {"message": "Hi in non existing shop"}


@app.post("/find_nearest_images_from_image")
async def find_nearest_images_from_image(file: UploadFile):
    file_stream = await file.read()
    if not check_is_type_available(BytesIO(file_stream)):
        raise HTTPException(
            status_code=400,
            detail="The file extension is unavailable"
        )

    img = Image.open(BytesIO(file_stream))
    embeddings_img = model_for_image.encode_images([img])
    result_img = query_collection(embeddings_img, configuration.collection_name, configuration.model_for_image_label, 1)
    file_stream_out = get_google_file(result_img[0])

    return StreamingResponse(iter([file_stream_out]), media_type="image/jpeg")


@app.post("/find_nearest_images_from_image_zip")
async def find_nearest_images_from_image_zip(file: UploadFile):
    file_stream = await file.read()
    if not check_is_type_available(BytesIO(file_stream)):
        raise HTTPException(
            status_code=400,
            detail="The file extension is unavailable"
        )

    embeddings_img = model_for_image.encode_images([Image.open(BytesIO(file_stream))])
    result_img = query_collection(embeddings_img, configuration.collection_name, configuration.model_for_image_label, 5)
    file_streams_out = [get_google_file(im) for im in result_img]
    zip_buffer = put_images_to_zip(file_streams_out)
    zip_buffer.seek(0)

    return StreamingResponse(zip_buffer, media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=images.zip"})


@app.post("/find_nearest_images_from_text")
async def find_nearest_images_from_text(text: str):
    if len(text) > configuration.max_text_length:
        raise HTTPException(
            status_code=400,
            detail="The input text is too long"
        )

    embeddings_query = model_for_text.encode_text(text)
    result_img = query_collection(embeddings_query, configuration.collection_name, configuration.model_for_text_label, 1)
    file_stream_out = get_google_file(result_img[0])
    return StreamingResponse(iter([file_stream_out]), media_type="image/jpeg")


@app.post("/find_nearest_images_from_text_zip")
async def find_nearest_images_from_text_zip(text: str):
    if len(text) > configuration.max_text_length:
        raise HTTPException(
            status_code=400,
            detail="The input text is too long"
        )

    embeddings_query = model_for_text.encode_text(text)
    result_img = query_collection(embeddings_query, configuration.collection_name, configuration.model_for_text_label, 5)
    file_streams_out = [get_google_file(im) for im in result_img]
    zip_buffer = put_images_to_zip(file_streams_out)
    zip_buffer.seek(0)

    return StreamingResponse(zip_buffer, media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=images.zip"})
