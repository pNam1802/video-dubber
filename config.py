import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'outputs'
    
    # Tạo thư mục nếu chưa có
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)