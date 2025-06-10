from flask import Flask, request, jsonify
import os
import tempfile
import subprocess
import numpy
app = Flask(__name__)

# Path to your dynamic analysis script
DYNAMIC_SCRIPT = "dynamic_analyze.py"
# Where to store uploaded APKs
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/dynamic_scan', methods=['POST'])
def dynamic_scan():
    if 'file' not in request.files:
        return jsonify({'error': 'No APK provided under key "file"'}), 400

    apk_file = request.files['file']
    if not apk_file.filename.lower().endswith('.apk'):
        return jsonify({'error': 'Uploaded file is not an APK'}), 400

    # Save APK to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix='.apk')
    apk_path = tmp.name
    apk_file.save(apk_path)
    tmp.close()

    try:
        # Run dynamic analysis script
        proc = subprocess.run(
            ['python', DYNAMIC_SCRIPT, apk_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600
        )
        if proc.returncode != 0:
            return jsonify({'error': 'Dynamic analysis failed', 'details': proc.stderr}), 500

        out = proc.stdout
        # Look for the prediction line
        # Expected format: "Prediction: [0]" or "Prediction: [1]"
        pred = None
        for line in out.splitlines():
            if line.strip().startswith('Prediction:'):
                # e.g. Prediction: [1]
                part = line.split(':', 1)[1].strip()
                pred = part.strip(' []')
                break

        if pred is None:
            return jsonify({'error': 'Could not parse prediction', 'output': out}), 500

        label = 'malware' if pred == '1' else 'benign'
        return jsonify({'prediction': label, 'raw_output': out})

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Dynamic analysis timed out'}), 500
    finally:
        try:
            os.remove(apk_path)
        except OSError:
            pass
import sys
print("Python executable:", sys.executable)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

