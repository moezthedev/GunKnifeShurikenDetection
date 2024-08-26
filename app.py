# app.py
from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_objects(image, threshold_value=90, min_area=500):
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_image, connectivity=8)

    objects = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        separate_object = np.zeros_like(image)
        separate_object[labels == label] = 255
        objects.append((separate_object, label))
    return objects, labels

def extract_sift_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_descriptors(descriptors1, descriptors2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return good_matches

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result_image_path = process_image(filepath)
        return render_template('result.html', result_image_path=result_image_path)

def process_image(image_path):
    template_images = {
        'gun': extract_sift_descriptors(cv2.imread('templates/gun.png', cv2.IMREAD_GRAYSCALE)),
        'knife': extract_sift_descriptors(cv2.imread('templates/knife.png', cv2.IMREAD_GRAYSCALE)),
        'shuriken': extract_sift_descriptors(cv2.imread('templates/shuriken.png', cv2.IMREAD_GRAYSCALE))
    }
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    extracted_objects, labels = extract_objects(image)
    mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    colors = {'gun': (255, 0, 0), 'knife': (0, 0, 255), 'shuriken': (0, 255, 255)}

    for extracted_object, label in extracted_objects:
        best_match = {'template': None, 'matches': 0}
        for template_name, (template_kp, template_desc) in template_images.items():
            region_kp, region_desc = extract_sift_descriptors(extracted_object)
            matches = match_descriptors(region_desc, template_desc)
            if len(matches) > best_match['matches']:
                best_match['template'] = template_name
                best_match['matches'] = len(matches)
        if best_match['matches'] > 10:
            mask[labels == label] = colors[best_match['template']]

    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
    cv2.imwrite(result_image_path, mask)
    return 'uploads/result.png'

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
