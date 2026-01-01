let stream = null;
let capturedSamples = [];

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

// Camera functions
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        const video = document.getElementById('video');
        video.srcObject = stream;
        capturedSamples = [];
        document.getElementById('samplesPreview').innerHTML = '';
    } catch (error) {
        alert('Error accessing camera: ' + error.message);
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    const video = document.getElementById('video');
    video.srcObject = null;
}

function captureSample() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    capturedSamples.push(imageData);
    
    // Display preview
    const preview = document.createElement('div');
    preview.className = 'sample-preview';
    preview.innerHTML = `<img src="${imageData}" alt="Sample ${capturedSamples.length}">`;
    document.getElementById('samplesPreview').appendChild(preview);
    
    // Show message
    const messageDiv = document.getElementById('registerMessage');
    messageDiv.innerHTML = `<div class="alert alert-info">Sample ${capturedSamples.length} captured</div>`;
}

// Register user
document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (capturedSamples.length < 1) {
        alert('Please capture at least one sample');
        return;
    }
    
    const name = document.getElementById('userName').value;
    const email = document.getElementById('userEmail').value;
    const notes = document.getElementById('userNotes').value;
    
    const messageDiv = document.getElementById('registerMessage');
    messageDiv.innerHTML = '<div class="alert alert-info">Registering user...</div>';
    
    try {
        const response = await apiFetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                email: email,
                notes: notes,
                images: capturedSamples
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            messageDiv.innerHTML = `<div class="alert alert-success">User registered successfully! User ID: ${data.user_id}</div>`;
            document.getElementById('registerForm').reset();
            capturedSamples = [];
            document.getElementById('samplesPreview').innerHTML = '';
            stopCamera();
            loadUsers(); // Refresh users list
        } else {
            messageDiv.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
        }
    } catch (error) {
        messageDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    }
});

// Load users
async function loadUsers() {
    try {
        const response = await apiFetch('/api/users');
        const data = await response.json();
        const tbody = document.getElementById('usersTableBody');
        
        if (data.users.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center">No users registered</td></tr>';
            return;
        }
        
        tbody.innerHTML = data.users.map(user => `
            <tr>
                <td>${user.user_id}</td>
                <td>${user.name}</td>
                <td>${user.email || '-'}</td>
                <td>${new Date(user.enrollment_date).toLocaleDateString()}</td>
                <td>
                    <button class="btn btn-sm btn-danger" onclick="deleteUser(${user.user_id})">Delete</button>
                </td>
            </tr>
        `).join('');
        
        // Update attendance user dropdown
        const select = document.getElementById('attendanceUserId');
        select.innerHTML = '<option value="">All Users</option>' + 
            data.users.map(user => `<option value="${user.user_id}">${user.name}</option>`).join('');
    } catch (error) {
        document.getElementById('usersTableBody').innerHTML = 
            '<tr><td colspan="5" class="text-center text-danger">Error loading users</td></tr>';
    }
}

// Delete user
async function deleteUser(userId) {
    if (!confirm('Are you sure you want to delete this user? This action cannot be undone.')) {
        return;
    }
    
    try {
        const response = await apiFetch(`/api/users/${userId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (response.ok) {
            alert('User deleted successfully');
            loadUsers();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

// Load attendance
async function loadAttendance() {
    const date = document.getElementById('attendanceDate').value;
    const userId = document.getElementById('attendanceUserId').value;
    
    if (!date) {
        document.getElementById('attendanceTableBody').innerHTML = 
            '<tr><td colspan="6" class="text-center">Please select a date</td></tr>';
        return;
    }
    
    try {
        let url = `/api/attendance?date=${date}`;
        if (userId) {
            url = `/api/attendance?user_id=${userId}`;
        }
        
        const response = await apiFetch(url);
        const data = await response.json();
        const tbody = document.getElementById('attendanceTableBody');
        
        if (data.attendance.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center">No attendance records found</td></tr>';
            return;
        }
        
        tbody.innerHTML = data.attendance.map(record => `
            <tr>
                <td>${record.name || 'N/A'}</td>
                <td>${record.date}</td>
                <td>${record.time}</td>
                <td><span class="badge bg-success">${record.status}</span></td>
                <td>${(record.confidence * 100).toFixed(1)}%</td>
                <td><span class="badge bg-info">${record.liveness_status || 'N/A'}</span></td>
            </tr>
        `).join('');
    } catch (error) {
        document.getElementById('attendanceTableBody').innerHTML = 
            '<tr><td colspan="6" class="text-center text-danger">Error loading attendance</td></tr>';
    }
}

// Set today's date as default
document.addEventListener('DOMContentLoaded', () => {
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('attendanceDate').value = today;
    loadUsers();
    loadAttendance();
});

