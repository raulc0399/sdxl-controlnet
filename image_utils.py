import datetime
import json
import os

class ImageUtils:
    @staticmethod
    def save_image_with_timestamp(image, suffix="generated", params=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{suffix}.jpg"
        folder = "./gen_imgs"
        os.makedirs(folder, exist_ok=True)  # Ensure the directory exists

        file_path = os.path.join(folder, filename)  # Create the full path for the file
        image.save(file_path)

        if params is not None:
            json_file_path = file_path.replace(".jpg", ".json")
            with open(json_file_path, "w") as json_file:
                json.dump(params, json_file)