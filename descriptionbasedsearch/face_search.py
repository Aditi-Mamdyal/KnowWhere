import os
import logging
from deepface import DeepFace

# Set up logging to catch errors without crashing the program
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceSearch")

# Paths aligned with your computer
BASE_DIR = os.path.dirname(__file__)
IMAGE_FOLDER = r"D:\coding\college"
FACE_DB_PATH = os.path.join(BASE_DIR, "face_database")

def search_face(query_name):
    """
    Enhanced face search with error handling and performance optimizations.
    """
    # --- FIX 1: Missing Database Handling ---
    if not os.path.exists(FACE_DB_PATH):
        logger.error(f"Critical Error: The root folder '{FACE_DB_PATH}' does not exist.")
        # Create it automatically so the program doesn't crash next time
        os.makedirs(FACE_DB_PATH, exist_ok=True)
        return []

    person_db_path = os.path.join(FACE_DB_PATH, query_name)

    # --- FIX 2: Specific Person/Empty Folder Handling ---
    if not os.path.exists(person_db_path):
        logger.warning(f"No entry found for '{query_name}' in face_database.")
        return []
    
    # Check if there are actually images in the folder
    valid_images = [f for f in os.listdir(person_db_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not valid_images:
        logger.warning(f"Folder for '{query_name}' exists but contains no valid images.")
        return []

    # Use the first valid image as the reference
    reference_img = os.path.join(person_db_path, valid_images[0])
    results = []

    # --- FIX 3: Handling the Scan Loop Shortcomings ---
    for root, dirs, files in os.walk(IMAGE_FOLDER):
        for file in files:
            # Shortcoming: Scanning non-images is a waste of CPU
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(root, file)

            try:
                # Shortcoming: DeepFace detection can be slow. 
                # We use 'opencv' backend here because it is the fastest for corporate/office environments.
                match = DeepFace.verify(
                    img1_path=reference_img,
                    img2_path=img_path,
                    enforce_detection=False, 
                    detector_backend='opencv', 
                    model_name="VGG-Face",
                    silent=True # Prevents console spam
                )

                if match["verified"]:
                    # We return the distance score (lower is better, but we flip it for consistency)
                    confidence = 1 - match["distance"]
                    results.append((confidence, img_path))

            except Exception as e:
                # Catching specific errors like 'Image not readable'
                logger.error(f"Could not process image {file}: {e}")
                continue

    # Sort results so the most likely match is at the top
    results.sort(key=lambda x: x[0], reverse=True)
    return results