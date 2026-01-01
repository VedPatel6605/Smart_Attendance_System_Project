let stream = null;
let verificationInterval = null;
let isVerifying = false;
let sessionId = null;
const VERIFICATION_INTERVAL_MS = 700;

const API_BASE_URL = (() => {
    if (window.API_BASE_URL) {
        return window.API_BASE_URL.replace(/\/$/, '');
    }
    if (window.location.origin && window.location.origin.startsWith('http')) {
        return window.location.origin.replace(/\/$/, '');
    }
    return 'http://localhost:5000';
})();

function buildApiUrl(path) {
    try {
        return new URL(path, API_BASE_URL).toString();
    } catch (error) {
        const normalizedPath = path.startsWith('/') ? path : `/${path}`;
        return `${API_BASE_URL}${normalizedPath}`;
    }
}

function apiFetch(path, options = {}) {
    const url = buildApiUrl(path);
    return fetch(url, options);
}

function generateSessionId() {
    if (window.crypto && window.crypto.randomUUID) {
        return window.crypto.randomUUID();
    }
    return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function startVerification() {
    if (isVerifying) return;
    
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        const video = document.getElementById('video');
        video.srcObject = stream;
        
        sessionId = generateSessionId();
        isVerifying = true;
        document.getElementById('statusDisplay').innerHTML = 
            '<p class="status-warning">Verifying... Please look at the camera and blink naturally.</p>';
        
        // Start continuous verification
        verificationInterval = setInterval(verifyFrame, VERIFICATION_INTERVAL_MS);
        
    } catch (error) {
        alert('Error accessing camera: ' + error.message);
    }
}

function stopVerification() {
    if (verificationInterval) {
        clearInterval(verificationInterval);
        verificationInterval = null;
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    const video = document.getElementById('video');
    video.srcObject = null;
    
    isVerifying = false;
    sessionId = null;
    document.getElementById('statusDisplay').innerHTML = 
        '<p class="text-muted">Verification stopped</p>';
    document.getElementById('resultDisplay').innerHTML = '';
}

async function verifyFrame() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
        return;
    }
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    try {
        const response = await apiFetch('/api/verify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                device_id: 'webcam',
                session_id: sessionId
            })
        });
        
        const data = await response.json();
        displayResult(data);
        
    } catch (error) {
        document.getElementById('statusDisplay').innerHTML = 
            `<p class="status-error">Error: ${error.message}</p>`;
    }
}

function displayResult(data) {
    const statusDiv = document.getElementById('statusDisplay');
    const resultDiv = document.getElementById('resultDisplay');
    
    if (data.stage === 'liveness' && data.liveness_info) {
        const statusClass = data.liveness === 'blink_detected' ? 'status-success' :
            data.liveness === 'no_face' || data.liveness === 'no_blink' ? 'status-error' : 'status-warning';
        const framesAnalyzed = data.liveness_info.frames_analyzed !== undefined
            ? `<p><strong>Frames analyzed:</strong> ${data.liveness_info.frames_analyzed}</p>`
            : '';
        statusDiv.innerHTML = `
            <p class="${statusClass}">${data.liveness_info.message || 'Perform the blink challenge'}</p>
            ${framesAnalyzed}
        `;
        if (data.liveness === 'no_face') {
            resultDiv.innerHTML = `
                <div class="result-card result-error">
                    <p><strong>Status:</strong> Face not detected consistently</p>
                    <p>Ensure you are centered, well lit, and avoid occlusions.</p>
                </div>
            `;
        } else if (data.liveness === 'no_blink') {
            resultDiv.innerHTML = `
                <div class="result-card result-error">
                    <p><strong>Status:</strong> Blink not detected</p>
                    <p>Please blink naturally a couple of times.</p>
                </div>
            `;
        } else {
            resultDiv.innerHTML = '';
        }
        return;
    }
    
    if (data.success) {
        statusDiv.innerHTML = `
            <p class="status-success">✓ Verification Successful!</p>
            <p><strong>User:</strong> ${data.name}</p>
            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
            <p><strong>Liveness:</strong> ${data.liveness}</p>
        `;
        
        resultDiv.innerHTML = `
            <div class="result-card result-success">
                <h6>Attendance Marked Successfully</h6>
                <p><strong>Name:</strong> ${data.name}</p>
                <p><strong>Time:</strong> ${new Date(data.timestamp).toLocaleString()}</p>
                <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                <p><strong>Attendance ID:</strong> ${data.attendance_id}</p>
            </div>
        `;
        
        // Stop verification after successful match
        setTimeout(() => {
            stopVerification();
        }, 3000);
        
    } else {
        statusDiv.innerHTML = `
            <p class="status-warning">Verifying... Please look at the camera</p>
        `;
        
        if (data.error) {
            resultDiv.innerHTML = `
                <div class="result-card result-error">
                    <p><strong>Status:</strong> ${data.error}</p>
                    ${data.confidence ? `<p><strong>Best Match:</strong> ${(data.confidence * 100).toFixed(1)}%</p>` : ''}
                    ${data.liveness_info ? `<p><strong>Details:</strong> ${data.liveness_info.message || ''}</p>` : ''}
                </div>
            `;
        }
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopVerification();
});

