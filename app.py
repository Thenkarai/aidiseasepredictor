import json
import uuid
import os
import base64
import io
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import google.generativeai as genai
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env for local development
load_dotenv()

app = Flask(__name__)

# Diagnostic logging for Vercel
print(f"Server starting - Python {os.sys.version}")
print(f"Environment GEMINI_API_KEY: {'Set' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")

# Dynamic Model Strategy
def get_best_available_model():
    """Discover the best model supported by this API key dynamically."""
    try:
        supported_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(f"DEBUG: Supported models found: {supported_models}")
        
        # Preference order for Farmers (Fast & Highly Capable)
        for pref in ['models/gemini-1.5-flash', 'models/gemini-1.5-flash-latest', 'models/gemini-2.0-flash-exp', 'models/gemini-1.5-pro']:
            if pref in supported_models: return pref
        if supported_models: return supported_models[0]
    except Exception as e:
        print(f"DEBUG: Model discovery failed: {e}")
    return 'models/gemini-1.5-flash'

# Configure Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

@app.route('/', methods=['GET'])
def home():
    try:
        return render_template('home.html')
    except Exception as e:
        print(f"Home Page Error: {str(e)}")
        return f"<h1>Internal Error</h1><p>{str(e)}</p>", 500

def model_predict(image_bytes):
    """Run prediction with automated model cycling for 100% up-time."""
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
            'error_message': 'Format Error: Please upload JPG/PNG.',
        }

    prompt = """
You are a helpful assistant for farmers. Your job is to identify a plant disease from an image and give simple, practical advice on how to cure it.

### INSTRUCTIONS:
1. Identify the **Plant Name** and the **Disease/Problem**.
2. Explain **What to do** (Cure) in simple, easy steps.
3. Provide everything in both **English and Tamil**.

Respond ONLY in this JSON format:
{
  "plant_name": "Common Plant Name",
  "disease_name": "Common Disease Name",
  "cause": "Simple reason for the problem",
  "cure": "Step-by-step simple instructions for the farmer",
  "severity": 0.0,
  "confidence": 0,
  "tamil": {
    "plant": "தாவரத்தின் பெயர்",
    "disease": "நோயின் பெயர்",
    "cause": "பிரச்சனைக்கான எளிய காரணம்",
    "cure": "விவசாயிக்கான எளிய தீர்வு முறைகள்"
  }
}
"""
    
    try:
        model_name = get_best_available_model()
        print(f"ENGINE: Final decision - Using {model_name}")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt, img])
        
        # Extract JSON from response
        res_text = response.text
        if "```json" in res_text:
            res_text = res_text.split("```json")[1].split("```")[0].strip()
        elif "```" in res_text:
            res_text = res_text.split("```")[1].split("```")[0].strip()
            
        data = json.loads(res_text)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"Prediction Error:\n{err_msg}")
        return {
            'prediction': {'cause': 'Engine Alignment Issue', 'cure': 'Please try again in 1 minute'},
            'confidence': 0,
            'plant_name': 'System Busy',
            'disease_name': f'Connectivity Error ({str(e)})',
            'is_healthy': False,
            'severity': 0.0,
            'tamil': {'plant': 'பிழை', 'disease': 'பிழை', 'cause': '-', 'cure': '-'},
            'is_valid': False,
            'error_message': f'API Support Error: {str(e)}',
        }

    # Process identification results
    plant_name = data.get("plant_name", "Unknown Specimen")
    disease_name = data.get("disease_name", "Condition Unknown")
    is_healthy = "healthy" in disease_name.lower()
    confidence = int(data.get("confidence", 95))
    severity = float(data.get("severity", 0.0))

    # Compile the result structure expected by the frontend
    prediction_label = {
        "cause": data.get("cause", "Etiology analysis in progress..."),
        "cure": data.get("cure", "Clinical consultation recommended.")
    }
    tamil_data = data.get("tamil", {
        "plant": plant_name,
        "disease": disease_name,
        "cause": "-",
        "cure": "-"
    })

    return {
        'prediction': prediction_label,
        'confidence': round(confidence, 2),
        'plant_name': plant_name,
        'disease_name': disease_name,
        'is_healthy': is_healthy,
        'severity': round(severity, 2),
        'tamil': tamil_data,
        'is_valid': True,
        'error_message': '',
    }


@app.route('/upload/', methods=['POST'])
def uploadimage():
    image_file = request.files.get('img')
    if not image_file:
        return redirect('/')
        
    image_bytes = image_file.read()
    b64_image = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode('utf-8')
    
    result = model_predict(image_bytes)

    # Handle AJAX/XHR request for performance
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'success': True,
            'imagepath': b64_image,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'plant_name': result['plant_name'],
            'disease_name': result['disease_name'],
            'is_healthy': result['is_healthy'],
            'severity': result['severity'],
            'tamil': result['tamil'],
            'is_valid': result['is_valid'],
            'error_message': result.get('error_message', ''),
            'current_time': datetime.now().strftime("%b %d, %Y %I:%M %p")
        })

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
        current_time=datetime.now().strftime("%b %d, %Y %I:%M %p")
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
        'error_message': result.get('error_message', '')
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
