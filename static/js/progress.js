let currentTaskId = null;
let selectedFile = null;

// Initialize drag and drop
document.addEventListener('DOMContentLoaded', function() {
    const uploadZone = document.getElementById('uploadZone');
    const videoInput = document.getElementById('videoInput');

    // Drag and drop handlers
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // File input change
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
});

function handleFileSelect(file) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.style.display = 'none';

    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime', 'video/webm'];
    if (!validTypes.includes(file.type)) {
        showError('Vui lòng chọn file video hợp lệ (MP4, AVI, MOV, WebM)');
        return;
    }

    // Validate file size (500MB)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File quá lớn! Vui lòng chọn file dưới 500MB');
        return;
    }

    // Store file and show info
    selectedFile = file;
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    
    fileName.textContent = file.name;
    fileSize.textContent = ` (${formatFileSize(file.size)})`;
    fileInfo.style.display = 'block';
    
    document.getElementById('processBtn').disabled = false;
}

function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    selectedFile = null;
    document.getElementById('processBtn').disabled = true;
    document.getElementById('fileInfo').style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function startProcessing() {
    if (!selectedFile) {
        showError('Vui lòng chọn file video!');
        return;
    }

    const formData = new FormData();
    formData.append('video', selectedFile);

    // Hide upload section, show progress
    document.getElementById('uploadSection').style.display = 'none';
    document.getElementById('progressSection').style.display = 'block';
    document.getElementById('statusText').textContent = 'Đang tải video lên...';
    document.getElementById('progressBar').style.width = '5%';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload thất bại! Vui lòng thử lại.');
        }

        const data = await response.json();
        currentTaskId = data.task_id;
        checkStatus();
    } catch (error) {
        showError('Có lỗi xảy ra: ' + error.message);
        document.getElementById('uploadSection').style.display = 'block';
        document.getElementById('progressSection').style.display = 'none';
    }
}

async function checkStatus() {
    if (!currentTaskId) return;

    try {
        const response = await fetch(`/status/${currentTaskId}`);
        if (!response.ok) {
            throw new Error('Không thể kiểm tra trạng thái');
        }

        const data = await response.json();
        
        if (data.state === 'SUCCESS') {
            // Show success with video player
            document.getElementById('progressSection').style.display = 'none';
            document.getElementById('resultSection').style.display = 'block';
            
            // Set video source and load
            const videoElement = document.getElementById('resultVideo');
            videoElement.src = `/download/${data.result_url}`;
            videoElement.load();
            
            // Try to get video duration
            videoElement.addEventListener('loadedmetadata', function() {
                const duration = videoElement.duration;
                const mins = Math.floor(duration / 60);
                const secs = Math.floor(duration % 60);
                document.getElementById('videoDuration').textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
            });
        } else if (data.state === 'FAILURE') {
            // Show error
            showError('Xử lý thất bại: ' + (data.error || 'Lỗi không xác định'));
            document.getElementById('uploadSection').style.display = 'block';
            document.getElementById('progressSection').style.display = 'none';
        } else {
            // Update progress (PENDING or PROGRESS)
            const progress = data.current || 0;
            const total = data.total || 100;
            const percentage = Math.max(5, Math.min(95, Math.round((progress / total) * 100)));
            
            document.getElementById('progressBar').style.width = percentage + '%';
            
            // Update status text
            let statusText = data.status || 'Đang xử lý...';
            if (data.current > 0 && data.total > 0) {
                statusText = `${statusText} (${data.current}/${data.total})`;
            }
            document.getElementById('statusText').textContent = statusText;
            
            // Continue checking
            setTimeout(checkStatus, 2000);
        }
    } catch (error) {
        console.error('Lỗi kiểm tra trạng thái:', error);
        // Continue trying
        setTimeout(checkStatus, 3000);
    }
}

// Video player controls
function togglePlayPause() {
    const video = document.getElementById('resultVideo');
    if (video.paused) {
        video.play();
    } else {
        video.pause();
    }
}

function downloadVideo() {
    const video = document.getElementById('resultVideo');
    const link = document.createElement('a');
    link.href = video.src;
    link.download = video.src.split('/').pop() || 'video.mp4';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function shareVideo() {
    const video = document.getElementById('resultVideo');
    const videoUrl = window.location.origin + video.src;
    
    if (navigator.share) {
        navigator.share({
            title: 'Video Dubber AI',
            text: 'Xem video được thuyết minh tiếng Việt!',
            url: videoUrl
        }).catch(err => console.log('Error sharing:', err));
    } else {
        // Fallback: copy to clipboard
        navigator.clipboard.writeText(videoUrl);
        alert('Link video đã được copy vào clipboard!');
    }
}