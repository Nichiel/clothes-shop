import magic
import zipfile

from io import BytesIO


def put_images_to_zip(file_streams_out):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for idx, file_stream_out in enumerate(file_streams_out):
            zip_file.writestr(f"image_{idx}.jpg", file_stream_out)

    return zip_buffer


def check_is_type_available(bytes_io) -> bool:
    mime = magic.Magic(mime=True)
    file_type = mime.from_buffer(bytes_io.getvalue())
    if file_type == 'image/jpeg':
        return True
    else:
        return False
