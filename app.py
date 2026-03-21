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

# Advanced Model Strategy
AVAILABLE_MODELS = ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-2.0-flash-exp', 'gemini-1.5-pro']

def get_available_models():
    """Diagnostic helper to log all models accessible by the current key."""
    try:
        models = [m.name for m in genai.list_models()]
        print(f"DIAGNOSTIC - Accessible Models: {models}")
        return models
    except Exception as e:
        print(f"DIAGNOSTIC - Failed to list models: {e}")
        return []

# Configure Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    get_available_models()

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
You are acting as the Universal Botanical Diagnostic Engine. Your mission is to provide an absolute, high-precision identification and pathological report for the provided image.

### CORE TASK:
Analyze the image with the depth of a world-class plant pathologist. Identify the **Plant Species** and the specific **Condition/Disease** with 100% coverage across all global crop varieties, ornamental plants, and medicinal flora.

### DATA EXTRACTION:
1. SPECIES IDENTIFICATION: Even if the leaf is highly diseased, identify the species using its cellular structure, vein patterns, and morphology.
2. PATHOLOGICAL ANALYSIS: Identify the specific disease, pest, or deficiency or state 'Healthy'.
3. ETIOLOGY: Describe the biological root cause (fungal, bacterial, viral, environmental).
4. CLINICAL CURE: Provide the most effective, professional-grade treatment or recovery plan.

### METRICS:
- Severity: Percentage of tissue damage (0.0 to 100.0).
- Confidence: Statistical certainty of this diagnosis (0 to 100).

### LOCALIZATION:
- Provide accurate Tamil equivalents for the Identification, Cause, and Cure.

Respond ONLY in this JSON format:
{
  "plant_name": "...",
  "disease_name": "...",
  "cause": "...",
  "cure": "...",
  "severity": 0.0,
  "confidence": 0,
  "tamil": {
    "plant": "...",
    "disease": "...",
    "cause": "...",
    "cure": "..."
  }
}
"""
    
    last_error = None
    for model_name in AVAILABLE_MODELS:
        try:
            print(f"ENGINE - Attempting analysis with: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, img])
            
            # Extract JSON from response
            res_text = response.text
            if "```json" in res_text:
                res_text = res_text.split("```json")[1].split("```")[0].strip()
            elif "```" in res_text:
                res_text = res_text.split("```")[1].split("```")[0].strip()
                
            data = json.loads(res_text)
            print(f"ENGINE - Success with {model_name}")
            break # Exit loop on success
        except Exception as e:
            last_error = str(e)
            print(f"ENGINE - Model {model_name} failed: {last_error}")
            continue # Try next model
    else:
        # If the loop finishes without break, all models failed
        return {
            'prediction': {'cause': 'All models returned errors', 'cure': 'Verify API Permissions'},
            'confidence': 0,
            'plant_name': 'Engine Error',
            'disease_name': f'Connectivity Issue ({last_error})',
            'is_healthy': False,
            'severity': 0.0,
            'tamil': {'plant': 'பிழை', 'disease': 'பிழை', 'cause': '-', 'cure': '-'},
            'is_valid': False,
            'error_message': f'Diagnostic Failure: {last_error}',
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
