// ===== Elements =====
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const webcamBtn = document.getElementById('webcamBtn');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const previewSection = document.getElementById('previewSection');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const downloadBtn = document.getElementById('downloadBtn');

let currentImage = null;
let stream = null;

// ===== Upload Area Events =====
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// ===== File Handling =====
function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImage = file;
        preview.src = e.target.result;
        previewSection.hidden = false;
        resultsSection.hidden = true;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.hidden = true;
        }
    };
    reader.readAsDataURL(file);
}

// ===== Webcam Handling =====
webcamBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        video.srcObject = stream;
        video.hidden = false;
        previewSection.hidden = true;
        resultsSection.hidden = true;
        
        // Change button to capture
        webcamBtn.textContent = '📸 Capture Fingerprint';
        webcamBtn.style.background = '#ef4444';
        webcamBtn.onclick = captureFromWebcam;
    } catch (error) {
        alert('Unable to access webcam: ' + error.message);
    }
});

function captureFromWebcam() {
    // Capture image from video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    const imageDataUrl = canvas.toDataURL('image/jpeg');
    preview.src = imageDataUrl;
    previewSection.hidden = false;
    
    // Stop webcam
    stream.getTracks().forEach(track => track.stop());
    video.hidden = true;
    
    // Reset button
    webcamBtn.textContent = '📷 Use Webcam';
    webcamBtn.style.background = '#10b981';
    webcamBtn.onclick = () => webcamBtn.click();
    
    // Store image for analysis
    currentImage = dataURLtoFile(imageDataUrl, 'webcam.jpg');
}

// ===== Analysis =====
analyzeBtn.addEventListener('click', async () => {
    if (!currentImage) return;
    
    // Show loading
    previewSection.hidden = true;
    loadingSection.hidden = false;
    
    // Prepare form data
    const formData = new FormData();
    
    if (currentImage instanceof File) {
        formData.append('file', currentImage);
    } else {
        formData.append('image', preview.src);
    }
    
    try {
        // Send to backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Wait for animation
            setTimeout(() => {
                displayResults(result);
            }, 2000);
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        alert('Analysis failed: ' + error.message);
        loadingSection.hidden = true;
        previewSection.hidden = false;
    }
});

// ===== Display Results =====
function displayResults(result) {
    loadingSection.hidden = true;
    resultsSection.hidden = false;
    
    // Blood group
    document.getElementById('bloodGroup').textContent = result.blood_group;
    
    // Confidence
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    confidenceFill.style.width = result.confidence + '%';
    confidenceText.textContent = result.confidence + '%';
    
    // Probability bars
    const probabilityBars = document.getElementById('probabilityBars');
    probabilityBars.innerHTML = '';
    
    const sortedProbs = Object.entries(result.all_probabilities)
        .sort((a, b) => b[1] - a[1]);
    
    sortedProbs.forEach(([bloodGroup, prob]) => {
        const barHtml = `
            <div class="prob-bar">
                <span class="prob-label">${bloodGroup}</span>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width: ${prob}%"></div>
                </div>
                <span class="prob-value">${prob.toFixed(1)}%</span>
            </div>
        `;
        probabilityBars.innerHTML += barHtml;
    });
}

// ===== New Analysis =====
newAnalysisBtn.addEventListener('click', () => {
    resultsSection.hidden = true;
    previewSection.hidden = true;
    currentImage = null;
    fileInput.value = '';
});

// ===== Download Report =====
downloadBtn.addEventListener('click', () => {
    const bloodGroup = document.getElementById('bloodGroup').textContent;
    const confidence = document.getElementById('confidenceText').textContent;
    
    const report = `
Blood Group Detection Report
=============================

Detected Blood Group: ${bloodGroup}
Confidence Level: ${confidence}
Analysis Date: ${new Date().toLocaleString()}

Model Accuracy: ${document.querySelector('.stat-value').textContent}

This report is generated by an AI-powered blood group detection system.
For medical purposes, please confirm with laboratory testing.
    `;
    
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `blood_group_report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
});

// ===== Helper Functions =====
function dataURLtoFile(dataurl, filename) {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
}
