import os
import sys
import whisper
import torch
import subprocess
from transformers import MarianMTModel, MarianTokenizer
import asyncio
import edge_tts
from config import Config
import re

# Set FFmpeg path
ffmpeg_path = r'C:\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe'
os.environ['PATH'] = ffmpeg_path.rsplit('\\', 1)[0] + ';' + os.environ.get('PATH', '')

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

def speedup_audio(input_audio, output_audio, tempo):
    """Tăng tốc audio bằng ffmpeg atempo filter (giữ pitch)"""
    try:
        cmd = [
            ffmpeg_path,
            '-i', input_audio,
            '-filter:a', f'atempo={tempo}',
            '-y',
            output_audio
        ]
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', timeout=60)
        if result.returncode != 0:
            print(f'[WARNING] FFmpeg speedup failed: {result.stderr}')
            return False
        return True
    except Exception as e:
        print(f'[WARNING] Error speedup audio: {e}')
        return False

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

def update_task_status(task_id, state, meta):
    """Cập nhật trạng thái task"""
    task_status[task_id] = {'state': state, 'meta': meta}

def dubbing_process(task_id, video_path, output_filename):
    """Xử lý dubbing video"""
    try:
        update_task_status(task_id, 'PROGRESS', {'status': 'Đang trích xuất văn bản (Whisper)...', 'current': 0, 'total': 100, 'error': None})
        
        # Load models
        whisper_model = get_whisper_model()
        translation_model, tokenizer = get_translation_model()
        device = get_device()
        
        # 1. Transcribe
        result = whisper_model.transcribe(video_path, verbose=False)
        raw_segments = result['segments']
        
        update_task_status(task_id, 'PROGRESS', {'status': f'Đã nhận diện {len(raw_segments)} đoạn, đang tối ưu...', 'current': 10, 'total': 100, 'error': None})
        
        # Optimize segments
        segments = filter_segments(raw_segments)
        segments = merge_segments(segments)
        
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
            
            # Dịch bằng MarianMT
            inputs = tokenizer(text_to_translate, return_tensors="pt", padding=True).to(device)
            translated_tokens = translation_model.generate(**inputs)
            translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            
            # Khôi phục từ khóa ML sau dịch
            translated_text = restore_ml_keywords(translated_text, ml_replacements)
            
            # Tạo file âm thanh tạm (WAV format - no compression)
            temp_audio = f"outputs/temp_{task_id}_{i}.wav"
            asyncio.run(generate_voice(translated_text, temp_audio))
            
            # Đo thời lượng TTS
            original_duration = end_time - start_time
            target_duration = original_duration * 0.95  # Buffer 5% để tránh overlap
            tts_duration = get_audio_duration(temp_audio)
            
            # Nếu TTS dài hơn target, tăng tốc
            if tts_duration and tts_duration > target_duration:
                tempo = target_duration / tts_duration
                # Giới hạn tempo trong khoảng hợp lý (0.8 - 2.0)
                tempo = max(0.8, min(2.0, tempo))
                
                print(f'[Task {task_id}] Segment {i}: TTS={tts_duration:.2f}s, Target={target_duration:.2f}s, Tempo={tempo:.2f}')
                
                # Speed up audio
                speedup_audio(temp_audio, temp_audio, tempo)
            
            audio_segments.append({
                'audio': temp_audio,
                'start': start_time,
                'end': end_time,
                'text': translated_text
            })

        update_task_status(task_id, 'PROGRESS', {'status': f'Đã tạo {len(audio_segments)} audio clips, đang ghép vào video...', 'current': 80, 'total': 100, 'error': None})

        # 3. Merge audio và video bằng ffmpeg
        output_path = os.path.join(Config.OUTPUT_FOLDER, output_filename)
        
        # Nếu không có audio segments, copy video gốc
        if len(audio_segments) == 0:
            import shutil
            shutil.copy2(video_path, output_path)
            update_task_status(task_id, 'SUCCESS', {'status': 'Hoàn tất!', 'current': 100, 'total': 100, 'error': None, 'result_url': output_filename})
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
        
        update_task_status(task_id, 'SUCCESS', {'status': 'Hoàn tất!', 'current': 100, 'total': 100, 'error': None, 'result_url': output_filename})
    except Exception as e:
        import traceback
        error_msg = f'{type(e).__name__}: {str(e)}'
        print(f'[ERROR Task {task_id}] {error_msg}')
        print(traceback.format_exc())
        update_task_status(task_id, 'FAILURE', {'status': f'Lỗi: {error_msg}', 'current': 0, 'total': 100, 'error': error_msg})