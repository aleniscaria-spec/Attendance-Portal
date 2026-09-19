import os
import logging
import numpy as np
import face_recognition

# Configure logging for face engine
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_engine")


class FaceRecognitionEngine:
    """
    Face Recognition Engine for Student Attendance Portal.

    Responsibilities:
    - Load student face images from the dataset/ directory.
    - Generate 128-dimensional face encodings using dlib via face-recognition.
    - Store encodings and corresponding student names in memory for fast lookup.
    - Validate registration photos (ensuring exactly one face is present).
    - Compare incoming browser camera frames against known encodings.
    - Reload database in memory when a new student is registered.

    NOTE ON FACE DISTANCE THRESHOLD:
    The face recognition model computes Euclidean distance between 128-d vectors.
    A distance of 0.0 means identical faces.
    A threshold around 0.50 (default) determines whether two faces match.
    We do NOT treat '1 - distance' as a mathematical probability or confidence score;
    we simply evaluate whether min_distance <= tolerance.
    """

    def __init__(self, dataset_dir="dataset", tolerance=0.50):
        """
        Initialize the face recognition engine.

        :param dataset_dir: Directory where student images are stored.
        :param tolerance: Distance threshold for considering a face match (default: 0.50).
        """
        self.dataset_dir = dataset_dir
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []

        # Ensure dataset directory exists
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir, exist_ok=True)
            logger.info(f"Created dataset directory at: {self.dataset_dir}")

        # Load existing student encodings into memory
        self.load_database()

    def load_database(self):
        """
        Scan the dataset directory and load all student face encodings into memory.
        Filename format: Firstname_Lastname.jpg -> Name: 'Firstname Lastname'
        """
        self.known_face_encodings = []
        self.known_face_names = []

        if not os.path.isdir(self.dataset_dir):
            logger.warning(f"Dataset path '{self.dataset_dir}' is not a directory.")
            return

        supported_extensions = (".jpg", ".jpeg", ".png")
        files = os.listdir(self.dataset_dir)

        logger.info(f"Scanning '{self.dataset_dir}' for student reference images...")

        for file_name in files:
            # Skip hidden files or files without image extensions
            if not file_name.lower().endswith(supported_extensions):
                continue

            image_path = os.path.join(self.dataset_dir, file_name)

            try:
                # Load image file into RGB numpy array
                student_image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(student_image)

                if len(encodings) > 0:
                    # Use the first detected face encoding
                    self.known_face_encodings.append(encodings[0])

                    # Convert filename like 'Alen_I_Scaria.jpg' to 'Alen I Scaria'
                    base_name, _ = os.path.splitext(file_name)
                    student_name = base_name.replace("_", " ").strip()
                    self.known_face_names.append(student_name)

                    logger.info(f"Loaded student face: {student_name} ({file_name})")
                else:
                    logger.warning(
                        f"No face detected in '{file_name}'. Skipping file."
                    )
            except Exception as e:
                logger.error(f"Error loading image '{file_name}': {e}")

        logger.info(
            f"Database loaded: {len(self.known_face_names)} student(s) ready."
        )

    def validate_registration_image(self, image_rgb):
        """
        Validate an image submitted for student registration.

        Criteria:
        - Exactly ONE face must be present.
        - Rejects image if no face exists.
        - Rejects image if multiple faces exist.

        :param image_rgb: Numpy array (RGB) representing the captured image.
        :return: (is_valid: bool, message: str, encoding: np.ndarray or None)
        """
        if image_rgb is None or not isinstance(image_rgb, np.ndarray):
            return False, "Invalid image data provided.", None

        try:
            face_locations = face_recognition.face_locations(image_rgb)
            face_count = len(face_locations)

            if face_count == 0:
                return (
                    False,
                    "No face detected. Please ensure your face is well-lit and directly facing the camera.",
                    None,
                )

            if face_count > 1:
                return (
                    False,
                    f"Multiple faces detected ({face_count}). Exactly one face is required for registration.",
                    None,
                )

            # Generate encoding for the single detected face
            encodings = face_recognition.face_encodings(image_rgb, face_locations)
            if not encodings:
                return False, "Unable to extract face features. Please try again.", None

            return True, "Face validated successfully.", encodings[0]

        except Exception as e:
            logger.error(f"Error during registration face validation: {e}")
            return False, f"Validation error: {str(e)}", None

    def recognize(self, image_rgb):
        """
        Recognize a face in the incoming browser camera frame.

        :param image_rgb: Numpy array (RGB) representing the camera frame.
        :return: dict with keys:
                 - status: 'recognized' | 'unknown' | 'no_face' | 'error'
                 - name: student name or 'Unknown' or None
                 - distance: minimum Euclidean distance found (float) or None
                 - message: human-readable status explanation
        """
        if image_rgb is None or not isinstance(image_rgb, np.ndarray):
            return {
                "status": "error",
                "name": None,
                "distance": None,
                "message": "Invalid image data received.",
            }

        try:
            # Detect face locations in the current frame
            face_locations = face_recognition.face_locations(image_rgb)

            if len(face_locations) == 0:
                return {
                    "status": "no_face",
                    "name": None,
                    "distance": None,
                    "message": "No face detected in the frame. Please look at the camera.",
                }

            # Check if we have registered students
            if len(self.known_face_encodings) == 0:
                return {
                    "status": "unknown",
                    "name": "Unknown",
                    "distance": None,
                    "message": "No registered students in the database.",
                }

            # Generate face encodings for detected faces
            face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
            if not face_encodings:
                return {
                    "status": "no_face",
                    "name": None,
                    "distance": None,
                    "message": "Could not extract face encoding.",
                }

            # Focus on the primary detected face
            current_encoding = face_encodings[0]

            # Calculate Euclidean distances to all known student face encodings
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, current_encoding
            )

            best_match_index = int(np.argmin(face_distances))
            min_distance = float(face_distances[best_match_index])

            # Check whether minimum distance is within the matching tolerance
            if min_distance <= self.tolerance:
                recognized_name = self.known_face_names[best_match_index]
                return {
                    "status": "recognized",
                    "name": recognized_name,
                    "distance": round(min_distance, 4),
                    "message": f"Student recognized: {recognized_name}",
                }
            else:
                return {
                    "status": "unknown",
                    "name": "Unknown",
                    "distance": round(min_distance, 4),
                    "message": "Face did not match any registered student.",
                }

        except Exception as e:
            logger.error(f"Error during face recognition: {e}")
            return {
                "status": "error",
                "name": None,
                "distance": None,
                "message": f"Recognition error: {str(e)}",
            }

    def reload(self):
        """
        Reload the face database from disk into memory.
        Called whenever a new student is registered.
        """
        logger.info("Reloading face recognition database...")
        self.load_database()

    def student_count(self):
        """Return the number of registered students."""
        return len(self.known_face_names)

    def student_names(self):
        """Return the list of all registered student names."""
        return list(self.known_face_names)
