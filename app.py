import json
import uuid
import os
import base64
import io
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
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



# The uploadimages route is no longer needed because images are handled in-memory.
# If needed, you can serve static files via Flask's static folder.



@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

def model_predict(image_bytes):
    """Run prediction and return disease info + confidence data using Gemini API."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
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
    
    return {
        'prediction': prediction_label,
        'confidence': round(confidence, 2),
        'plant_name': plant_name,
        'disease_name': disease_name,
        'is_healthy': is_healthy,
        'severity': 0.0,
        'tamil': tamil_data,
        'is_valid': True,
        'error_message': '',
        'error_tamil': '',
    }


@app.route('/upload/', methods=['POST'])
def uploadimage():
    image_file = request.files.get('img')
    if not image_file:
        return redirect('/')
        
    image_bytes = image_file.read()
    b64_image = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode('utf-8')
    
    result = model_predict(image_bytes)

    return render_template(
        'home.html',
        result=True,
        imagepath=b64_image,
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


@app.route('/upload-camera/', methods=['POST'])
def upload_camera():
    """Handle camera capture — receives base64 image data."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    image_data = data['image']
    b64_string = image_data
    if ',' in image_data:
        b64_string = image_data.split(',')[1]

    img_bytes = base64.b64decode(b64_string)

    result = model_predict(img_bytes)

    return jsonify({
        'success': True,
        'imagepath': image_data,
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

@app.route('/plants', methods=['GET'])
def get_plants():
    """Return the full plant catalog as JSON."""
    catalog_path = os.path.join(os.path.dirname(__file__), 'plants_catalog.json')
    with open(catalog_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
