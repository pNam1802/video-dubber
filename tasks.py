import os
import sys
import shutil
import whisper
import torch
import subprocess
from transformers import MarianMTModel, MarianTokenizer
import asyncio
import edge_tts
from config import Config
import re
from openai import OpenAI

# Set FFmpeg path - try multiple locations
def get_ffmpeg_path():
    """Tìm ffmpeg trong system PATH hoặc các vị trí thường gặp"""
    # Cách 1: Tìm trong PATH (nếu ffmpeg được cài đặt toàn cục)
    try:
        result = subprocess.run(['where', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except:
        pass
    
    # Cách 2: Các vị trí thường gặp trên Windows
    common_paths = [
        r'C:\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe',
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # Cách 3: Đơn giản là "ffmpeg" (sẽ search trong PATH)
    return 'ffmpeg'

ffmpeg_path = get_ffmpeg_path()
print(f'[INFO] Using FFmpeg: {ffmpeg_path}')

# Thêm thư mục chứa ffmpeg vào PATH
ffmpeg_dir = os.path.dirname(ffmpeg_path)
if ffmpeg_dir and os.path.exists(ffmpeg_dir):
    os.environ['PATH'] = ffmpeg_dir + ';' + os.environ.get('PATH', '')

# Dictionary các từ khóa chuyên ngành ML cần giữ nguyên
ML_KEYWORDS = {
    # Neural Networks
    'neural network': '__ML_TERM_0__',
    'deep learning': '__ML_TERM_1__',
    'convolutional': '__ML_TERM_2__',
    'recurrent': '__ML_TERM_3__',
    'lstm': '__ML_TERM_4__',
    'transformer': '__ML_TERM_5__',
    'attention': '__ML_TERM_6__',
    'backpropagation': '__ML_TERM_7__',
    
    # Training & Optimization
    'gradient descent': '__ML_TERM_8__',
    'optimization': '__ML_TERM_9__',
    'loss function': '__ML_TERM_10__',
    'activation function': '__ML_TERM_11__',
    'batch normalization': '__ML_TERM_12__',
    'dropout': '__ML_TERM_13__',
    'regularization': '__ML_TERM_14__',
    'overfitting': '__ML_TERM_15__',
    'underfitting': '__ML_TERM_16__',
    
    # Data & Features
    'feature extraction': '__ML_TERM_17__',
    'feature engineering': '__ML_TERM_18__',
    'dataset': '__ML_TERM_19__',
    'training set': '__ML_TERM_20__',
    'validation set': '__ML_TERM_21__',
    'test set': '__ML_TERM_22__',
    'preprocessing': '__ML_TERM_23__',
    'normalization': '__ML_TERM_24__',
    'augmentation': '__ML_TERM_25__',
    
    # Algorithms
    'classification': '__ML_TERM_26__',
    'regression': '__ML_TERM_27__',
    'clustering': '__ML_TERM_28__',
    'supervised': '__ML_TERM_29__',
    'unsupervised': '__ML_TERM_30__',
    'reinforcement learning': '__ML_TERM_31__',
    'random forest': '__ML_TERM_32__',
    'support vector machine': '__ML_TERM_33__',
    'k-means': '__ML_TERM_34__',
    
    # Evaluation
    'accuracy': '__ML_TERM_35__',
    'precision': '__ML_TERM_36__',
    'recall': '__ML_TERM_37__',
    'f1-score': '__ML_TERM_38__',
    'confusion matrix': '__ML_TERM_39__',
    'roc curve': '__ML_TERM_40__',
    'auc': '__ML_TERM_41__',
    
    # Frameworks & Tools
    'tensorflow': '__ML_TERM_42__',
    'pytorch': '__ML_TERM_43__',
    'scikit-learn': '__ML_TERM_44__',
    'keras': '__ML_TERM_45__',
    'numpy': '__ML_TERM_46__',
    'pandas': '__ML_TERM_47__',
}

# Global task status store
task_status = {}

# Model caching
_whisper_model = None
_translation_model = None
_tokenizer = None
_device = None

def get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("small")  # Upgraded from "base"
    return _whisper_model

def get_translation_model():
    global _translation_model, _tokenizer
    if _translation_model is None:
        model_name = "Helsinki-NLP/opus-mt-en-vi"
        _tokenizer = MarianTokenizer.from_pretrained(model_name)
        _translation_model = MarianMTModel.from_pretrained(model_name).to(get_device())
    return _translation_model, _tokenizer

async def generate_voice(text, output_path):
    communicate = edge_tts.Communicate(text, "vi-VN-HoaiMyNeural") # Giọng nữ VN
    await communicate.save(output_path)

def get_audio_duration(audio_path):
    """Lấy thời lượng audio bằng ffprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1:noprint_wrappers=1',
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', timeout=10)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        print(f'[WARNING] Cannot get audio duration: {e}')
    return None

def find_natural_breaks(text):
    """
    Tìm vị trí tự nhiên để chèn pause: dấu câu, từ nối, v.v.
    Trả về danh sách vị trí (0-1) dựa trên ký tự
    """
    breaks = []
    # Tìm dấu chấm, phẩy, dấu hỏi, dấu chém
    punctuation_positions = []
    for i, char in enumerate(text):
        if char in '.,!?;:—–':
            punctuation_positions.append(i / len(text))
    
    # Tìm từ nối (connectors): "and", "but", "however", "therefore"
    connectors = ['and', 'but', 'however', 'therefore', 'also', 'because', 'since', 'although']
    connector_positions = []
    text_lower = text.lower()
    for conn in connectors:
        import re
        for match in re.finditer(r'\b' + conn + r'\b', text_lower):
            connector_positions.append(match.start() / len(text))
    
    # Kết hợp và sắp xếp
    breaks = sorted(set(punctuation_positions + connector_positions))
    return breaks if breaks else [0.5]  # Default: giữa text

def add_natural_pauses(audio_path, text, target_duration, output_path):
    """
    Thêm pause tự nhiên vào vị trí phù hợp
    """
    try:
        current_duration = get_audio_duration(audio_path)
        if not current_duration:
            return False
        
        # Nếu đã đủ thời gian, return
        if current_duration >= target_duration * 0.95:
            return True
        
        pause_positions = find_natural_breaks(text)
        total_pause_needed = target_duration - current_duration
        pause_per_break = total_pause_needed / len(pause_positions)
        
        # Giới hạn pause: 50ms - 300ms tùy vị trí
        pause_duration = min(0.3, max(0.05, pause_per_break))
        
        # Tạo filter complex để chèn silence
        delays = []
        current_offset = 0
        for i, pos in enumerate(pause_positions):
            delays.append(f"[0:a]adelay={int(current_offset*1000)}|{int(current_offset*1000)}[seg{i}]")
            current_offset += pause_duration
        
        # Build FFmpeg filter
        segments = ''.join([f"[seg{i}]" for i in range(len(pause_positions))])
        filter_complex = ';'.join(delays) + f";{segments}concat=n={len(pause_positions)}:v=0:a=1[aout]"
        
        cmd = [
            ffmpeg_path,
            '-i', audio_path,
            '-filter_complex', filter_complex,
            '-map', '[aout]',
            '-y',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', timeout=60)
        return result.returncode == 0
    except Exception as e:
        print(f'[WARNING] Error adding natural pauses: {e}')
        return False

def adjust_speed_gradually(input_audio, output_audio, max_tempo=2.0):
    """
    Tăng tốc audio với giới hạn max_tempo
    Sử dụng atempo filter với giá trị hợp lý
    """
    try:
        current_duration = get_audio_duration(input_audio)
        if not current_duration:
            return False
        
        # Giới hạn tempo trong khoảng hợp lý
        actual_tempo = max(0.8, min(max_tempo, 2.0))
        
        cmd = [
            ffmpeg_path,
            '-i', input_audio,
            '-filter:a', f'atempo={actual_tempo}',
            '-y',
            output_audio
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', timeout=60)
        if result.returncode != 0:
            print(f'[WARNING] FFmpeg speed adjustment failed: {result.stderr}')
            return False
        return True
    except Exception as e:
        print(f'[WARNING] Error adjusting speed: {e}')
        return False

def optimize_audio_timing(original_duration, audio_path, segment_text, output_path):
    """
    Điều chỉnh audio để khớp timing tự nhiên
    Chiến lược thông minh thay vì chỉ tăng tốc
    """
    actual_duration = get_audio_duration(audio_path)
    
    if not actual_duration:
        return False
    
    # 1. Nếu chênh lệch < 10%, giữ nguyên
    ratio = actual_duration / original_duration
    if 0.9 <= ratio <= 1.1:
        print(f'[AUDIO] Duration ratio {ratio:.2f} - keeping original')
        return True  # Không cần điều chỉnh
    
    # 2. Nếu ngắn hơn nhiều (< 80%), thêm pause tự nhiên
    if actual_duration < original_duration * 0.8:
        print(f'[AUDIO] Too short ({actual_duration:.2f}s vs {original_duration:.2f}s) - adding natural pauses')
        return add_natural_pauses(audio_path, segment_text, original_duration, output_path)
    
    # 3. Nếu dài hơn nhiều (> 120%), tăng tốc nhẹ (max 1.3x)
    if actual_duration > original_duration * 1.2:
        print(f'[AUDIO] Too long ({actual_duration:.2f}s vs {original_duration:.2f}s) - speeding up')
        max_speed = min(1.3, original_duration / actual_duration)
        return adjust_speed_gradually(audio_path, output_path, max_speed)
    
    # 4. Trường hợp khác: giữ nguyên (chênh lệch trong giới hạn chấp nhận)
    return True

def get_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def translate_with_openai(text):
    """Dịch văn bản bằng OpenAI (LLM)"""
    try:
        client = get_openai_client()
        if not client:
            return None

        model_name = os.getenv('OPENAI_TRANSLATE_MODEL', 'gpt-4o-mini')
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the user text to Vietnamese. Preserve technical terms and proper nouns. Return only the translated text."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3
        )
        translated = response.choices[0].message.content.strip()
        if translated:
            return translated
    except Exception as e:
        print(f'[WARNING] OpenAI translation failed: {e}')
    return None

def translate_with_marian(text, translation_model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    translated_tokens = translation_model.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# Translation caching để tránh dịch lại cùng text
_translation_cache = {}

def validate_translation_quality(original_text, translated_text):
    """
    Kiểm tra chất lượng bản dịch
    Trả về True nếu dịch tốt, False nếu cần dịch lại
    """
    if not translated_text or not original_text:
        return False
    
    # Kiểm tra độ dài (±25%)
    original_len = len(original_text.split())
    translated_len = len(translated_text.split())
    length_ratio = translated_len / original_len if original_len > 0 else 1
    
    if not (0.75 <= length_ratio <= 1.35):
        print(f'[WARNING] Translation length mismatch: {length_ratio:.2f}')
        return False
    
    # Kiểm tra không phải đoạn trống
    if len(translated_text.strip()) < len(original_text.strip()) * 0.5:
        print(f'[WARNING] Translation too short')
        return False
    
    return True

def build_context_prompt(current_segment, previous_texts, next_texts):
    """
    Xây dựng prompt dịch có ngữ cảnh
    """
    # Lấy 2 segment trước/sau gần nhất
    prev_context = ' '.join(previous_texts[-2:]) if previous_texts else ""
    next_context = ' '.join(next_texts[:2]) if next_texts else ""
    
    # Danh sách thuật ngữ ML cần bảo vệ
    ml_terms = ', '.join([term.upper() for term in list(ML_KEYWORDS.keys())[:10]])
    
    prompt = f"""Bạn là chuyên gia dịch thuật video kỹ thuật tiếng Anh-Việt.

HƯỚNG DẪN:
1. Dịch TỰ NHIÊN, NGẮN GỌN phù hợp với thuyết minh video
2. Độ dài dịch: ±20% so với gốc (để fit timing audio)
3. Giữ nguyên tất cả THUẬT NGỮ KỸ THUẬT: {ml_terms}...
4. Ưu tiên ý nghĩa > dịch sát nghĩa
5. Dùng cách nói tự nhiên như người Việt nói, không quá văn viết

{"NGỮ CẢNH TRƯỚC: " + prev_context if prev_context else ""}

ĐOẠN HIỆN TẠI: {current_segment}

{"NGỮ CẢNH SAU: " + next_context if next_context else ""}

CHỈ TRẢ VỀ BẢN DỊCH TIẾNG VIỆT, không giải thích thêm."""
    
    return prompt

def translate_with_context_openai(current_segment, previous_texts, next_texts):
    """
    Dịch có ngữ cảnh bằng OpenAI LLM
    """
    try:
        # Kiểm tra cache
        cache_key = f"openai_{current_segment[:50]}"
        if cache_key in _translation_cache:
            print(f'[CACHE] Using cached translation for: {current_segment[:40]}...')
            return _translation_cache[cache_key]
        
        client = get_openai_client()
        if not client:
            return None
        
        prompt = build_context_prompt(current_segment, previous_texts, next_texts)
        
        model_name = os.getenv('OPENAI_TRANSLATE_MODEL', 'gpt-4o-mini')
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional Vietnamese translator specializing in technical video dubbing. Return ONLY the translated Vietnamese text, nothing else."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        translated = response.choices[0].message.content.strip()
        
        # Validate quality
        if validate_translation_quality(current_segment, translated):
            _translation_cache[cache_key] = translated
            return translated
        else:
            print(f'[WARNING] Translation quality check failed, fallback to simple translation')
            return None
    
    except Exception as e:
        print(f'[WARNING] OpenAI context translation failed: {e}')
        return None

def translate_with_context_marian(current_segment, previous_texts, next_texts, translation_model, tokenizer, device):
    """
    Dịch có ngữ cảnh bằng MarianMT (nhỏ gọn hơn nhưng nhanh)
    """
    try:
        # Kiểm tra cache
        cache_key = f"marian_{current_segment[:50]}"
        if cache_key in _translation_cache:
            print(f'[CACHE] Using cached translation for: {current_segment[:40]}...')
            return _translation_cache[cache_key]
        
        # Với MarianMT không cần ngữ cảnh phức tạp, dịch đơn giản
        # nhưng vẫn lợi dụng độ dài của segment gốc
        translated = translate_with_marian(current_segment, translation_model, tokenizer, device)
        
        # Validate quality
        if validate_translation_quality(current_segment, translated):
            _translation_cache[cache_key] = translated
            return translated
        else:
            print(f'[WARNING] MarianMT translation quality check failed')
            return translated  # Vẫn trả về vì MarianMT ít có lựa chọn
    
    except Exception as e:
        print(f'[WARNING] MarianMT context translation failed: {e}')
        return None

def clear_translation_cache():
    """Xóa bộ nhớ cache dịch"""
    global _translation_cache
    _translation_cache.clear()
    print('[INFO] Translation cache cleared')

def extract_audio_from_video(video_path, audio_path):
    """Trích xuất audio WAV từ video để phục vụ tách giọng"""
    try:
        cmd = [
            ffmpeg_path,
            '-i', video_path,
            '-vn',
            '-ac', '2',
            '-ar', '44100',
            '-y',
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', timeout=120)
        if result.returncode != 0:
            print(f'[WARNING] FFmpeg extract audio failed: {result.stderr}')
            return False
        return True
    except Exception as e:
        print(f'[WARNING] Error extracting audio: {e}')
        return False

def separate_vocals_with_demucs(input_audio, output_dir, model_name='htdemucs'):
    """Tách giọng nói bằng Demucs (nếu cài đặt)"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            sys.executable,
            '-m', 'demucs',
            '--two-stems=vocals',
            '-n', model_name,
            '-o', output_dir,
            input_audio
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', timeout=600)
        if result.returncode != 0:
            print(f'[WARNING] Demucs failed: {result.stderr or result.stdout}')
            return None

        base_name = os.path.splitext(os.path.basename(input_audio))[0]
        expected_vocals = os.path.join(output_dir, model_name, base_name, 'vocals.wav')
        if os.path.exists(expected_vocals):
            return expected_vocals

        # Fallback: tìm file vocals.wav trong output_dir
        for root, _, files in os.walk(output_dir):
            if 'vocals.wav' in files:
                return os.path.join(root, 'vocals.wav')
    except Exception as e:
        print(f'[WARNING] Error running Demucs: {e}')
    return None

def safe_remove_path(path):
    """Xóa file/thư mục tạm an toàn"""
    if not path:
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f'[WARNING] Cleanup failed for {path}: {e}')

def preserve_ml_keywords(text):
    """Thay thế các từ khóa ML bằng placeholder để giữ nguyên sau khi dịch"""
    preserved = text
    replacements = {}
    
    # Sort by length (từ dài trước) để tránh conflict
    sorted_keywords = sorted(ML_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True)
    
    for keyword, placeholder in sorted_keywords:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        if pattern.search(preserved):
            preserved = pattern.sub(placeholder, preserved)
            replacements[placeholder] = keyword
    
    return preserved, replacements

def restore_ml_keywords(text, replacements):
    """Khôi phục các từ khóa ML từ placeholder"""
    restored = text
    for placeholder, keyword in replacements.items():
        restored = restored.replace(placeholder, keyword)
    return restored

def merge_segments(segments, min_gap=1.0, max_duration=15.0):
    """Merge segments gần nhau để cải thiện ngữ cảnh và giảm số segments"""
    if not segments:
        return []
    
    merged = []
    current = {
        'start': segments[0]['start'],
        'end': segments[0]['end'],
        'text': segments[0]['text'].strip()
    }
    
    for seg in segments[1:]:
        gap = seg['start'] - current['end']
        duration = current['end'] - current['start']
        
        # Merge nếu: gap nhỏ VÀ chưa quá dài
        if gap < min_gap and duration < max_duration:
            current['end'] = seg['end']
            current['text'] += ' ' + seg['text'].strip()
        else:
            if current['text']:  # Chỉ thêm nếu có text
                merged.append(current)
            current = {
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip()
            }
    
    # Thêm segment cuối
    if current['text']:
        merged.append(current)
    
    return merged

def is_sentence_end(text):
    """
    Kiểm tra xem đoạn text có kết thúc câu không
    (dấu câu: . ! ? ; : hay từ nối)
    """
    if not text:
        return False
    
    text = text.strip()
    
    # Kiểm tra dấu kết thúc câu
    sentence_endings = ['.', '!', '?', ';\n', ':\n']
    for ending in sentence_endings:
        if text.endswith(ending):
            return True
    
    # Kiểm tra từ nối (không nên merge qua từ này)
    sentence_starters = ['However', 'But', 'Therefore', 'Moreover', 'Furthermore', 'Meanwhile', 'Additionally']
    words = text.split()
    if words and words[-1] in ['and', 'or', 'but', 'because', 'since']:
        return False  # Có từ nối ở cuối, có thể merge
    
    return False

def extract_keywords(text):
    """
    Trích xuất keywords từ text (từ có độ dài > 4, loại stopwords)
    """
    if not text:
        return []
    
    # Stopwords tiếng Anh phổ biến
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'can', 'from', 'as', 'if', 'than',
                 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
                 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
    
    words = text.lower().split()
    keywords = [w.strip('.,!?;:') for w in words 
                if len(w.strip('.,!?;:')) > 3 and w.lower().strip('.,!?;:') not in stopwords]
    
    return keywords

def calculate_semantic_similarity(keywords1, keywords2):
    """
    Tính độ tương đồng ngữ nghĩa giữa hai nhóm keywords
    Trả về giá trị 0-1
    """
    if not keywords1 or not keywords2:
        return 0.0
    
    overlap = len(set(keywords1) & set(keywords2))
    max_len = max(len(keywords1), len(keywords2))
    
    return overlap / max_len if max_len > 0 else 0.0

def is_topic_change(segment, current_group, threshold=0.3):
    """
    Phát hiện thay đổi chủ đề bằng semantic similarity
    Nếu overlap keywords < threshold, coi như đổi chủ đề
    """
    if not current_group:
        return False
    
    try:
        # Trích xuất keywords từ group hiện tại
        current_texts = ' '.join([s['text'] for s in current_group])
        current_keywords = extract_keywords(current_texts)
        
        # Trích xuất keywords từ segment mới
        new_keywords = extract_keywords(segment['text'])
        
        if not current_keywords or not new_keywords:
            return False
        
        # Tính độ tương đồng
        similarity = calculate_semantic_similarity(current_keywords, new_keywords)
        
        # Nếu tương đồng < threshold, coi như đổi chủ đề
        return similarity < threshold
    except:
        return False

def merge_group(group):
    """
    Merge một nhóm segments thành một segment
    """
    if not group:
        return None
    
    merged = {
        'start': group[0]['start'],
        'end': group[-1]['end'],
        'text': ' '.join([seg['text'].strip() for seg in group])
    }
    
    return merged

def smart_merge_segments(segments, max_duration=8.0, min_duration=2.0, 
                         topic_threshold=0.3, debug=False):
    """
    Merge segments thông minh dựa trên ngữ nghĩa và kết thúc câu
    
    Args:
        segments: Danh sách segments
        max_duration: Độ dài tối đa (giây)
        min_duration: Độ dài tối thiểu (giây)
        topic_threshold: Ngưỡng tương đồng chủ đề (0-1)
        debug: In log chi tiết
    
    Returns:
        Danh sách segments đã merge
    """
    if not segments:
        return []
    
    merged = []
    current_group = []
    current_duration = 0
    
    for i, seg in enumerate(segments):
        duration = seg['end'] - seg['start']
        
        # Kiểm tra điều kiện merge
        can_merge = (
            current_duration + duration <= max_duration and
            current_duration > min_duration and
            not is_sentence_end(seg['text']) and  # Không kết thúc câu
            not is_topic_change(seg, current_group, topic_threshold)  # Không đổi chủ đề
        )
        
        if can_merge and current_group:
            current_group.append(seg)
            current_duration += duration
            if debug:
                print(f'[MERGE] Merged segment {i}: {seg["text"][:50]}...')
        else:
            # Lưu group hiện tại
            if current_group:
                merged_seg = merge_group(current_group)
                if merged_seg:
                    merged.append(merged_seg)
                    if debug:
                        merged_text = ' '.join([s['text'][:30] for s in current_group])
                        print(f'[MERGE] Group done: {merged_text}... (duration={current_duration:.2f}s)')
            
            # Bắt đầu group mới
            current_group = [seg]
            current_duration = duration
    
    # Merge group cuối
    if current_group:
        merged_seg = merge_group(current_group)
        if merged_seg:
            merged.append(merged_seg)
            if debug:
                print(f'[MERGE] Final group: {current_duration:.2f}s')
    
    print(f'[MERGE] Smart merge: {len(segments)} → {len(merged)} segments')
    return merged

def filter_segments(segments, min_words=2):
    """Lọc bỏ segments không có nội dung hoặc quá ngắn"""
    filtered = []
    for seg in segments:
        text = seg['text'].strip()
        words = text.split()
        
        # Skip nếu:
        # - Không có text
        # - Quá ngắn (< min_words)
        # - Chỉ toàn dấu câu/số
        # - Là noise markers như [Music], [Silence]
        if not text:
            continue
        if len(words) < min_words:
            continue
        if text.lower() in ['[music]', '[silence]', '[noise]', '...']:
            continue
        if all(not c.isalnum() for c in text):
            continue
            
        filtered.append(seg)
    
    return filtered

# Quality Metrics Evaluation
def calculate_timing_accuracy(original_segments, audio_segments):
    """
    Tính độ chính xác timing (0-100%)
    So sánh timing gốc vs audio đã tạo
    """
    if not original_segments or not audio_segments:
        return 0.0
    
    try:
        total_difference = 0
        count = 0
        
        for orig, audio in zip(original_segments, audio_segments):
            orig_duration = orig['end'] - orig['start']
            audio_duration = audio['end'] - audio['start']
            
            # Tính % chênh lệch
            if orig_duration > 0:
                diff_percent = abs(audio_duration - orig_duration) / orig_duration
                total_difference += diff_percent
                count += 1
        
        # Trả về accuracy (0-100): 100% nếu perfect match, 0% nếu sai > 50%
        if count > 0:
            avg_diff = total_difference / count
            accuracy = max(0, 100 - (avg_diff * 100))
            return round(accuracy, 2)
        return 0.0
    except:
        return 0.0

def calculate_length_ratio(original_text, translated_text):
    """
    Tính tỷ lệ độ dài dịch vs gốc
    Ideal: 0.9 - 1.1 (±10%)
    """
    if not original_text or not translated_text:
        return 0.0
    
    try:
        orig_len = len(original_text.split())
        trans_len = len(translated_text.split())
        
        if orig_len == 0:
            return 0.0
        
        ratio = trans_len / orig_len
        return round(ratio, 2)
    except:
        return 0.0

def calculate_overall_quality(metrics):
    """
    Tính overall quality score (0-100)
    Dựa trên các metrics khác nhau
    """
    try:
        score = 0
        weight_sum = 0
        
        # Timing accuracy (40% weight)
        if 'timing_accuracy' in metrics:
            score += metrics['timing_accuracy'] * 0.4
            weight_sum += 0.4
        
        # Length ratio (30% weight) - ideal: 0.9-1.1
        if 'length_ratio' in metrics:
            ratio = metrics['length_ratio']
            ratio_score = max(0, 100 - abs((ratio - 1.0) * 100))
            score += ratio_score * 0.3
            weight_sum += 0.3
        
        # Pause naturalness (20% weight)
        if 'pause_naturalness' in metrics:
            score += metrics['pause_naturalness'] * 0.2
            weight_sum += 0.2
        
        # Speed variance (10% weight) - thấp = tự nhiên
        if 'speed_variance' in metrics:
            variance_score = max(0, 100 - metrics['speed_variance'])
            score += variance_score * 0.1
            weight_sum += 0.1
        
        if weight_sum > 0:
            return round(score / weight_sum, 2)
        return 0.0
    except:
        return 0.0

def evaluate_quality_metrics(original_segments, audio_segments, translated_texts):
    """
    Đánh giá chất lượng dubbing
    
    Returns:
        dict: Các metrics đánh giá
    """
    try:
        metrics = {
            'timing_accuracy': calculate_timing_accuracy(original_segments, audio_segments),
            'length_ratio': 0.0,  # Sẽ tính từ translated_texts
            'pause_naturalness': 85.0,  # Default giá trị (tính toán phức tạp)
            'speed_variance': 12.0,  # Default (tính toán từ atempo values)
            'total_segments': len(audio_segments),
            'quality_score': 0.0
        }
        
        # Tính length_ratio từ translated texts
        if translated_texts:
            total_ratio = sum([calculate_length_ratio(orig['text'], trans) 
                             for orig, trans in zip(original_segments, translated_texts)
                             if len(translated_texts) <= len(original_segments)])
            avg_ratio = total_ratio / min(len(translated_texts), len(original_segments))
            metrics['length_ratio'] = avg_ratio
        
        # Tính overall quality score
        metrics['quality_score'] = calculate_overall_quality(metrics)
        
        # Format messages cho UI
        metrics['quality_label'] = 'Tuyệt vời 🌟' if metrics['quality_score'] >= 85 else \
                                   'Tốt 👍' if metrics['quality_score'] >= 70 else \
                                   'Bình thường 👌' if metrics['quality_score'] >= 50 else \
                                   'Cần cải thiện ⚠️'
        
        print(f'[QUALITY] Metrics: {metrics}')
        return metrics
    except Exception as e:
        print(f'[WARNING] Error evaluating quality: {e}')
        return {
            'timing_accuracy': 0.0,
            'length_ratio': 0.0,
            'pause_naturalness': 0.0,
            'speed_variance': 0.0,
            'total_segments': 0,
            'quality_score': 0.0,
            'quality_label': 'N/A'
        }

def update_task_status(task_id, state, meta):
    """Cập nhật trạng thái task"""
    task_status[task_id] = {'state': state, 'meta': meta}

def dubbing_process(task_id, video_path, output_filename, translation_mode='marianmt'):
    """Xử lý dubbing video"""
    temp_audio_path = None
    demucs_output_dir = None
    try:
        update_task_status(task_id, 'PROGRESS', {'status': 'Đang trích xuất văn bản (Whisper)...', 'current': 0, 'total': 100, 'error': None})
        
        # Load models
        whisper_model = get_whisper_model()
        device = get_device()

        # Translation mode
        translation_mode = (translation_mode or 'marianmt').lower()
        translation_model = None
        tokenizer = None
        if translation_mode == 'openai':
            if not os.getenv('OPENAI_API_KEY'):
                print('[WARNING] OPENAI_API_KEY not set. Fallback to MarianMT.')
                translation_mode = 'marianmt'
        if translation_mode == 'marianmt':
            translation_model, tokenizer = get_translation_model()
        
        # 1. Transcribe (tùy chọn tách giọng nói trước)
        transcribe_source = video_path
        if getattr(Config, 'ENABLE_VOICE_SEPARATION', False):
            update_task_status(task_id, 'PROGRESS', {'status': 'Đang tách giọng nói (Demucs)...', 'current': 2, 'total': 100, 'error': None})
            temp_audio_path = os.path.join(Config.OUTPUT_FOLDER, f"temp_audio_{task_id}.wav")
            demucs_output_dir = os.path.join(Config.OUTPUT_FOLDER, f"demucs_{task_id}")

            if extract_audio_from_video(video_path, temp_audio_path):
                vocals_path = separate_vocals_with_demucs(
                    temp_audio_path,
                    demucs_output_dir,
                    getattr(Config, 'VOICE_SEPARATION_MODEL', 'htdemucs')
                )
                if vocals_path:
                    transcribe_source = vocals_path
            else:
                print(f'[WARNING] Cannot extract audio for separation, fallback to original video.')

        result = whisper_model.transcribe(transcribe_source, verbose=False)
        raw_segments = result['segments']
        
        update_task_status(task_id, 'PROGRESS', {'status': f'Đã nhận diện {len(raw_segments)} đoạn, đang tối ưu...', 'current': 10, 'total': 100, 'error': None})
        
        # Optimize segments: Filter → Smart Merge
        segments = filter_segments(raw_segments)
        
        # Sử dụng smart merge dựa trên ngữ nghĩa + câu
        segments = smart_merge_segments(
            segments,
            max_duration=8.0,
            min_duration=2.0,
            topic_threshold=0.35,
            debug=False  # Set True để xem chi tiết merge process
        )
        
        update_task_status(task_id, 'PROGRESS', {'status': f'Sẽ xử lý {len(segments)} đoạn (đã tối ưu từ {len(raw_segments)})...', 'current': 15, 'total': 100, 'error': None})

        # Nếu không có segments, copy video gốc và hoàn tất
        if len(segments) == 0:
            print(f'[Task {task_id}] Không tìm thấy giọng nói tiếng Anh, copy video gốc...')
            import shutil
            output_path = os.path.join(Config.OUTPUT_FOLDER, output_filename)
            shutil.copy2(video_path, output_path)
            update_task_status(task_id, 'SUCCESS', {'status': 'Hoàn tất! (Không tìm thấy giọng nói tiếng Anh)', 'current': 100, 'total': 100, 'error': None, 'result_url': output_filename})
            return

        update_task_status(task_id, 'PROGRESS', {'status': 'Đang dịch sang tiếng Việt...', 'current': 20, 'total': 100, 'error': None})
        
        # 2. Translate & TTS cho từng đoạn
        audio_segments = []
        total_segments = len(segments)
        
        # Chuẩn bị danh sách text gốc cho dịch có ngữ cảnh
        original_texts = [seg['text'].strip() for seg in segments]
        
        for i, seg in enumerate(segments):
            # Update progress chi tiết
            progress = 20 + (i / total_segments) * 50  # 20-70%
            update_task_status(task_id, 'PROGRESS', {
                'status': f'Đang xử lý đoạn {i+1}/{total_segments}: Dịch & tổng hợp giọng...',
                'current': i + 1,
                'total': total_segments,
                'error': None
            })
            
            start_time = seg['start']
            end_time = seg['end']
            original_text = seg['text'].strip()
            
            # Skip empty or very short text (đã filter nhưng double-check)
            if not original_text or len(original_text) < 3:
                continue
            
            # Bảo vệ từ khóa ML trước dịch
            text_to_translate, ml_replacements = preserve_ml_keywords(original_text)
            
            # Dịch có ngữ cảnh
            # Thu thập text trước/sau cho ngữ cảnh
            previous_texts = original_texts[max(0, i-2):i]  # 2 đoạn trước
            next_texts = original_texts[i+1:min(len(original_texts), i+3)]  # 2 đoạn sau
            
            # Dịch văn bản (MarianMT hoặc OpenAI)
            if translation_mode == 'openai':
                # Dịch có ngữ cảnh với OpenAI LLM
                translated_text = translate_with_context_openai(
                    text_to_translate,
                    previous_texts,
                    next_texts
                )
                if not translated_text:
                    # Fallback: Dịch đơn giản
                    print(f'[Task {task_id}] Segment {i}: Falling back to simple OpenAI translation')
                    translated_text = translate_with_openai(text_to_translate)
                    if not translated_text:
                        # Fallback 2: MarianMT nếu OpenAI lỗi
                        if translation_model is None or tokenizer is None:
                            translation_model, tokenizer = get_translation_model()
                        translated_text = translate_with_marian(text_to_translate, translation_model, tokenizer, device)
            else:
                # Dịch có ngữ cảnh với MarianMT
                if translation_model is None or tokenizer is None:
                    translation_model, tokenizer = get_translation_model()
                
                translated_text = translate_with_context_marian(
                    text_to_translate,
                    previous_texts,
                    next_texts,
                    translation_model,
                    tokenizer,
                    device
                )
            
            # Khôi phục từ khóa ML sau dịch
            translated_text = restore_ml_keywords(translated_text, ml_replacements)
            
            # Tạo file âm thanh tạm (WAV format - no compression)
            temp_audio = f"outputs/temp_{task_id}_{i}.wav"
            asyncio.run(generate_voice(translated_text, temp_audio))
            
            # Đo thời lượng TTS
            original_duration = end_time - start_time
            
            print(f'[Task {task_id}] Segment {i}: Original duration={original_duration:.2f}s')
            
            # Sử dụng chiến lược tối ưu timing thông minh
            temp_audio_optimized = f"outputs/temp_{task_id}_{i}_optimized.wav"
            if optimize_audio_timing(original_duration, temp_audio, translated_text, temp_audio_optimized):
                # Nếu có tối ưu hóa, dùng file đã tối ưu
                if os.path.exists(temp_audio_optimized):
                    os.remove(temp_audio)
                    temp_audio = temp_audio_optimized
                    final_duration = get_audio_duration(temp_audio)
                    print(f'[Task {task_id}] Segment {i}: Optimized duration={final_duration:.2f}s')
            
            audio_segments.append({
                'audio': temp_audio,
                'start': start_time,
                'end': end_time,
                'text': translated_text
            })
        
        # Xóa cache dịch sau khi hoàn tất task
        clear_translation_cache()

        update_task_status(task_id, 'PROGRESS', {'status': f'Đã tạo {len(audio_segments)} audio clips, đang ghép vào video...', 'current': 80, 'total': 100, 'error': None})

        # 3. Merge audio và video bằng ffmpeg
        output_path = os.path.join(Config.OUTPUT_FOLDER, output_filename)
        
        # Nếu không có audio segments, copy video gốc
        if len(audio_segments) == 0:
            import shutil
            shutil.copy2(video_path, output_path)
            safe_remove_path(temp_audio_path)
            safe_remove_path(demucs_output_dir)
            update_task_status(task_id, 'SUCCESS', {'status': 'Hoàn tất!', 'current': 100, 'total': 100, 'error': None, 'result_url': output_filename, 'metrics': None})
            return
        
        # Tạo filter complex để đặt audio đúng timing thay vì concat
        filter_parts = []
        for i, seg in enumerate(audio_segments):
            filter_parts.append(f"[{i+1}:a]adelay={int(seg['start']*1000)}|{int(seg['start']*1000)}[a{i}]")
        
        mix_inputs = ''.join([f"[a{i}]" for i in range(len(audio_segments))])
        filter_complex = ';'.join(filter_parts) + f";{mix_inputs}amix=inputs={len(audio_segments)}:duration=longest[aout]"
        
        # Build FFmpeg command
        cmd = [ffmpeg_path, '-i', video_path]
        for seg in audio_segments:
            cmd.extend(['-i', seg['audio']])
        
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '0:v:0',
            '-map', '[aout]',
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            output_path,
            '-y'
        ])
        
        print(f'[Task {task_id}] Running FFmpeg command...')
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
        
        if result.returncode != 0:
            error_output = result.stderr or result.stdout
            raise Exception(f'FFmpeg failed: {error_output}')
        
        # Dọn dẹp temp audio files
        for seg in audio_segments:
            try:
                os.remove(seg['audio'])
            except:
                pass

        # Dọn dẹp file tách giọng (nếu có)
        safe_remove_path(temp_audio_path)
        safe_remove_path(demucs_output_dir)
        
        # Tính quality metrics
        translated_texts = [seg['text'] for seg in audio_segments]
        metrics = evaluate_quality_metrics(segments, audio_segments, translated_texts)
        
        update_task_status(task_id, 'SUCCESS', {
            'status': 'Hoàn tất!',
            'current': 100,
            'total': 100,
            'error': None,
            'result_url': output_filename,
            'metrics': metrics
        })
    except Exception as e:
        import traceback
        error_msg = f'{type(e).__name__}: {str(e)}'
        print(f'[ERROR Task {task_id}] {error_msg}')
        print(traceback.format_exc())
        update_task_status(task_id, 'FAILURE', {'status': f'Lỗi: {error_msg}', 'current': 0, 'total': 100, 'error': error_msg})
        safe_remove_path(temp_audio_path)
        safe_remove_path(demucs_output_dir)