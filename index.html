from flask import Flask, request, jsonify
from flask_cors import CORS
import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app, origins="*")

print("Loading face swap model... please wait...")
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(
    '/app/inswapper_128.onnx',
    download=False,
    download_zip=False
)
print("Model loaded! Ready to swap faces.")

def decode_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

@app.route('/swap', methods=['POST', 'OPTIONS'])
def swap():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.json
        target_img = decode_image(data['target'])
        face_img = decode_image(data['face'])

        face_faces = face_app.get(face_img)
        if len(face_faces) == 0:
            return jsonify({'error': 'No face detected in face photo'}), 400
        source_face = face_faces[0]

        target_faces = face_app.get(target_img)
        if len(target_faces) == 0:
            return jsonify({'error': 'No face detected in target image'}), 400

        result = target_img.copy()
        for target_face in target_faces:
            result = swapper.get(
                result,
                target_face,
                source_face,
                paste_back=True
            )

        return jsonify({'result': encode_image(result)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
