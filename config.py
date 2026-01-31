import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'outputs'
    
    # Voice separation (Demucs)
    ENABLE_VOICE_SEPARATION = True
    VOICE_SEPARATION_MODEL = 'htdemucs'
    
    # Tạo thư mục nếu chưa có
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)