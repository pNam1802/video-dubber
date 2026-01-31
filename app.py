from flask import Flask, render_template, request, jsonify, send_from_directory
from config import Config
from tasks import dubbing_process, task_status, update_task_status
import os
import uuid
import threading

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config.from_object(Config)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video part"}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        output_filename = "dubbed_" + filename
        task_id = str(uuid.uuid4())

        translator_mode = request.form.get('translator_mode', 'marianmt')
        
        # Khởi tạo trạng thái task
        update_task_status(task_id, 'PENDING', {'status': 'Đang chờ...', 'current': 0, 'total': 100, 'error': None})
        
        # Chạy xử lý trong thread riêng
        thread = threading.Thread(target=dubbing_process, args=(task_id, filepath, output_filename, translator_mode))
        thread.daemon = True
        thread.start()
        
        return jsonify({"task_id": task_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status/<task_id>')
def task_status_route(task_id):
    if task_id not in task_status:
        return jsonify({'state': 'PENDING', 'current': 0, 'total': 100, 'status': 'Đang khởi tạo...', 'error': None, 'metrics': None}), 200
    
    task_info = task_status[task_id]
    state = task_info['state']
    meta = task_info['meta']
    
    response = {
        'state': state,
        'current': meta.get('current', 0),
        'total': meta.get('total', 100),
        'status': meta.get('status', ''),
        'error': meta.get('error', None),
        'metrics': meta.get('metrics', None)
    }
    
    if state == 'SUCCESS':
        response['result_url'] = meta.get('result_url')
    
    return jsonify(response)

@app.route('/download/<filename>')
def download(filename):
    """Serve video file for viewing or downloading"""
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)