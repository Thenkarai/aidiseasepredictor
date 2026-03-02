from flask import Flask, render_template, request, redirect, send_from_directory, url_for, jsonify
import numpy as np
import json
import uuid
import os
import base64
import cv2
from google import genai
from google.genai import types
from PIL import Image

app = Flask(__name__)

# Configure Gemini Client
# We won't initialize it globally to allow the app to start even if the key isn't set yet.
client = None


# Supported plants the model can identify
# Empty now since we use Gemini and support ALL plants 
SUPPORTED_PLANTS = []

CONFIDENCE_THRESHOLD = 40  # Below this = not recognized


def analyze_disease_severity(image_path):
    """
    Analyze the actual disease-affected area of the leaf using image processing.
    Uses HSV color segmentation to detect healthy green tissue vs.
    diseased tissue (brown, yellow, black spots, lesions).
    Returns the percentage of leaf area that is affected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step 1: Isolate the leaf from background
    # Detect leaf pixels (non-white, non-very-dark background)
    lower_leaf = np.array([0, 20, 30])
    upper_leaf = np.array([180, 255, 245])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    # Remove very bright white/grey background
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    not_bg = cv2.inRange(gray, 15, 240)
    leaf_mask = cv2.bitwise_and(leaf_mask, not_bg)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    if total_leaf_pixels < 100:
        return 0.0

    # Step 2: Detect healthy green areas
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_on_leaf = cv2.bitwise_and(green_mask, leaf_mask)
    healthy_pixels = cv2.countNonZero(green_on_leaf)

    # Step 3: Detect diseased areas (brown, yellow, black spots, lesions)
    # Brown/tan regions
    lower_brown = np.array([8, 30, 30])
    upper_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Dark spots/lesions (very dark on leaf)
    lower_dark = np.array([0, 0, 15])
    upper_dark = np.array([180, 255, 60])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # Yellow/chlorosis
    lower_yellow = np.array([18, 40, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # White/powdery mildew areas
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Combine all diseased masks
    diseased_mask = cv2.bitwise_or(brown_mask, dark_mask)
    diseased_mask = cv2.bitwise_or(diseased_mask, yellow_mask)
    diseased_mask = cv2.bitwise_or(diseased_mask, white_mask)

    # Only count diseased pixels that are on the leaf
    diseased_on_leaf = cv2.bitwise_and(diseased_mask, leaf_mask)
    diseased_pixels = cv2.countNonZero(diseased_on_leaf)

    # Calculate percentage
    affected_pct = (diseased_pixels / total_leaf_pixels) * 100
    affected_pct = min(affected_pct, 100.0)

    return round(affected_pct, 1)


@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

def model_predict(image_path):
    """Run prediction and return disease info + confidence data using Gemini API."""
    try:
        img = Image.open(image_path)
    except Exception as e:
        return {
            'prediction': None,
            'confidence': 0,
            'plant_name': '',
            'disease_name': '',
            'is_healthy': False,
            'severity': 0.0,
            'tamil': {'plant': '', 'disease': '', 'cause': '', 'cure': ''},
            'is_valid': False,
            'error_message': 'Uploaded image file is corrupted or unsupported. Please use JPG/PNG.',
            'error_tamil': 'பதிவேற்றப்பட்ட படம் சேதமடைந்துள்ளது அல்லது ஆதரிக்கப்படவில்லை. தயவுசெய்து JPG/PNG படத்தை போடுங்க.',
        }

    prompt = """
You are an expert botanist and plant pathologist. Examine the provided image.
1. Determine if the image contains a plant, leaf, crop, or flower. If the image is NOT of a plant/leaf/crop/flower (e.g., a car, person, animal, random object), return "is_plant": false and leave the rest empty.
2. If it IS a plant, identify the specific plant name (e.g., Apple, Tomato).
3. Identify the disease name if any, or "Healthy" if the plant appears healthy.
4. Provide a brief 1-sentence cause of the disease (if healthy, leave empty).
5. Provide a brief 1-sentence treatment or cure for the disease (if healthy, leave empty).
6. State your confidence level as an integer out of 100.
7. Provide Tamil translations for the plant name, disease name ("ஆரோக்கியமான" if healthy), cause, and cure (leave cause/cure empty if healthy).

Respond strictly in the following JSON template:
{
  "is_plant": true/false,
  "plant_name": "...",
  "disease_name": "...",
  "cause": "...",
  "cure": "...",
  "confidence": 95,
  "tamil": {
    "plant": "...",
    "disease": "...",
    "cause": "...",
    "cure": "..."
  }
}
"""
    try:
        api_key = "AIzaSyAi8rFVHxehwgYvCW6tIJFrurQshwgfXyY"
        
        global client
        if client is None:
            client = genai.Client(api_key=api_key)
            
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        data = json.loads(response.text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'prediction': None,
            'confidence': 0,
            'plant_name': '',
            'disease_name': '',
            'is_healthy': False,
            'severity': 0.0,
            'tamil': {'plant': '', 'disease': '', 'cause': '', 'cure': ''},
            'is_valid': False,
            'error_message': 'Failed to process image with AI model. Please ensure GEMINI_API_KEY is correctly set.',
            'error_tamil': 'AI மாடல் மூலம் படத்தை சரிபார்க்க முடியவில்லை. GEMINI_API_KEY உள்ளதா என சோதிக்கவும்.',
        }

    if not data.get("is_plant", False):
        return {
            'prediction': None,
            'confidence': 0,
            'plant_name': '',
            'disease_name': '',
            'is_healthy': False,
            'severity': 0.0,
            'tamil': {'plant': '', 'disease': '', 'cause': '', 'cure': ''},
            'is_valid': False,
            'error_message': 'This image does not appear to contain a plant. Please upload a clear photo of a plant or leaf.',
            'error_tamil': 'இந்த படத்தில செடி இருக்க மாதிரி தெரியல. தயவுசெய்து ஒரு செடியின் படத்தை போடுங்க.',
        }
    
    plant_name = data.get("plant_name", "Unknown Plant")
    disease_name = data.get("disease_name", "Unknown Disease")
    is_healthy = disease_name.lower() == "healthy"
    confidence = float(data.get("confidence", 95))
    
    prediction_label = {"cause": data.get("cause", ""), "cure": data.get("cure", "")}
    tamil_data = data.get("tamil", {"plant": plant_name, "disease": disease_name, "cause": "", "cure": ""})
    
    # Real disease severity from image analysis using OpenCV
    severity = 0.0
    if not is_healthy:
        severity = analyze_disease_severity(image_path)
        
    return {
        'prediction': prediction_label,
        'confidence': round(confidence, 2),
        'plant_name': plant_name,
        'disease_name': disease_name,
        'is_healthy': is_healthy,
        'severity': severity,
        'tamil': tamil_data,
        'is_valid': True,
        'error_message': '',
        'error_tamil': '',
    }


@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        filepath = f'{temp_name}_{image.filename}'
        image.save(filepath)

        result = model_predict(f'./{filepath}')

        return render_template(
            'home.html',
            result=True,
            imagepath=f'/{filepath}',
            prediction=result['prediction'],
            confidence=result['confidence'],
            plant_name=result['plant_name'],
            disease_name=result['disease_name'],
            is_healthy=result['is_healthy'],
            severity=result['severity'],
            tamil=result['tamil'],
            is_valid=result['is_valid'],
            error_message=result['error_message'],
            error_tamil=result['error_tamil'],
            supported_plants=SUPPORTED_PLANTS,
        )
    else:
        return redirect('/')


@app.route('/upload-camera/', methods=['POST'])
def upload_camera():
    """Handle camera capture — receives base64 image data."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    image_data = data['image']
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    img_bytes = base64.b64decode(image_data)
    filepath = f"uploadimages/camera_{uuid.uuid4().hex}.jpg"

    with open(filepath, 'wb') as f:
        f.write(img_bytes)

    result = model_predict(f'./{filepath}')

    return jsonify({
        'success': True,
        'imagepath': f'/{filepath}',
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'plant_name': result['plant_name'],
        'disease_name': result['disease_name'],
        'is_healthy': result['is_healthy'],
        'severity': result['severity'],
        'tamil': result['tamil'],
        'is_valid': result['is_valid'],
        'error_message': result['error_message'],
        'error_tamil': result['error_tamil'],
        'supported_plants': SUPPORTED_PLANTS,
    })


if __name__ == "__main__":
    app.run(debug=True)
