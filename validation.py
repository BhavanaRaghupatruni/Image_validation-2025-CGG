import os  # validation of size, fullscan, sunglasses, mask, blur
import cv2
import numpy as np
from skimage.filters import sobel, roberts, laplace
import pickle
from skimage.feature import hog

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_edges = sobel(gray)
    roberts_edges = roberts(gray)
    laplacian_edges = laplace(gray)

    features = []
    for edges in [sobel_edges, roberts_edges, laplacian_edges]:
        features.extend([np.mean(edges), np.max(edges), np.var(edges)])
    return features

def check_image_size(image_path, min_size_kb, max_size_kb):
    try:
        size = os.path.getsize(image_path)
        size_kb = size / 1024
        size_valid = min_size_kb <= size_kb <= max_size_kb
        return size_kb, size_valid
    except Exception as e:
        return 0, False

def find_face_border_distances_and_percentage(image, min_fullscan, max_fullscan):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = haar_cascade_face.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) != 1:
        return "invalid_fullscan", 0

    x, y, w, h = faces[0]
    height, width, _ = image.shape
    face_area = w * h
    image_area = width * height
    face_percentage = (face_area / image_area) * 100

    if min_fullscan <= face_percentage <= max_fullscan:
        return "valid_fullscan", face_percentage
    else:
        return "invalid_fullscan", face_percentage

def detect_sunglasses(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return "Face angle is not correct"
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 0:
                return "sunglasses_detected"
    return "no_sunglasses_detected"

def detect_face_mask(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(rgb_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return "no_face"

    results = []
    for (x, y, w, h) in faces:
        face = rgb_image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 128))  
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        features, _ = hog(gray_face, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

        svm_model_filename = 'svm_mask_classifier.pkl'
        svm_model = pickle.load(open(svm_model_filename, 'rb'))

        face_features = np.array(features).reshape(1, -1)
        prediction = svm_model.predict(face_features)

        results.append('Mask' if prediction[0] == 0 else 'No Mask')

    return results

def is_blurry(image):
    model_filename = 'blur_image_detection_svm_model.pkl'
    loaded_model = pickle.load(open(model_filename, 'rb'))

    features = extract_features(image)
    features = np.array(features).reshape(1, -1)

    prediction = loaded_model.predict(features)[0]
    return prediction == 1

def validate_image(image_path, min_size_kb, max_size_kb, min_fullscan, max_fullscan):
    try:
        # Check file extension
        if not allowed_file(image_path):
            return {"status": "error", "reason": f"Invalid file type. Allowed types: {ALLOWED_EXTENSIONS}"}

        # Check file size
        size_kb, size_valid = check_image_size(image_path, min_size_kb, max_size_kb)
        if not size_valid:
            return {"status": "error", "reason": f"File size is {size_kb}KB. Allowed range: {min_size_kb}-{max_size_kb}KB"}

        # Read image
        image = cv2.imread(image_path)

        # Check face scan percentage
        fullscan_status, fullscan_percentage = find_face_border_distances_and_percentage(image, min_fullscan, max_fullscan)
        if fullscan_status == "invalid_fullscan":
            return {"status": "error", "reason": f"Face scan percentage is too low: {fullscan_percentage}%."}

        # Check for sunglasses
        sunglasses_status = detect_sunglasses(image)
        if sunglasses_status == "sunglasses_detected":
            return {"status": "error", "reason": "Sunglasses detected in the image."}

        # Check for mask
        mask_status = detect_face_mask(image)
        if mask_status == "no_face":
            return {"status": "error", "reason": "No face detected in the image."}
        if "Mask" in mask_status:
            return {"status": "error", "reason": "Mask detected in the image."}

        # Check if the image is blurry
        if is_blurry(image):
            return {"status": "error", "reason": "Image is blurry."}

        return {"status": "success"}

    except Exception as e:
        return {"status": "error", "reason": str(e)}
