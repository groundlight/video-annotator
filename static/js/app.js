// Global state
let currentJobId = null;
let currentProjectDir = null;
let progressInterval = null;
let uploadedFileJobId = null;

// DOM elements
const projectSelect = document.getElementById('project-select');
const loadProjectBtn = document.getElementById('load-project-btn');
const createNewProjectBtn = document.getElementById('create-new-project-btn');
const uploadSection = document.getElementById('upload-section');
const uploadArea = document.getElementById('upload-area');
const videoFileInput = document.getElementById('video-file');
const uploadSuccess = document.getElementById('upload-success');
const uploadedFilename = document.getElementById('uploaded-filename');
const removeFileBtn = document.getElementById('remove-file-btn');
const setupSection = document.getElementById('setup-section');
const trainingSection = document.getElementById('training-section');
const productionSection = document.getElementById('production-section');
const progressSection = document.getElementById('progress-section');
const resultsSection = document.getElementById('results-section');
const errorMessage = document.getElementById('error-message');
const errorText = document.getElementById('error-text');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadProjects();
});

// Project selection
createNewProjectBtn.addEventListener('click', () => {
    uploadSection.style.display = 'block';
    setupSection.style.display = 'none';
    trainingSection.style.display = 'none';
    productionSection.style.display = 'none';
    hideError();
});

loadProjectBtn.addEventListener('click', () => {
    const selectedProject = projectSelect.value;
    if (!selectedProject) {
        showError('Please select a project');
        return;
    }
    
    // Load project details
    const projects = JSON.parse(projectSelect.dataset.projects || '[]');
    const project = projects.find(p => p.project_dir === selectedProject);
    
    if (project) {
        currentProjectDir = project.project_dir;
        uploadSection.style.display = 'none';
        setupSection.style.display = 'none';
        
        // Show training and production sections if project is set up
        if (project.frame_count > 0) {
            trainingSection.style.display = 'block';
            productionSection.style.display = 'block';
        } else {
            showError('Project has not been set up yet. Please run setup first.');
        }
    }
});

// File upload
uploadArea.addEventListener('click', () => videoFileInput.click());

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
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

videoFileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

removeFileBtn.addEventListener('click', () => {
    uploadedFileJobId = null;
    uploadSuccess.style.display = 'none';
    videoFileInput.value = '';
    setupSection.style.display = 'none';
});

async function handleFileSelect(file) {
    hideError();
    
    // Validate file
    const allowedExts = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedExts.includes(fileExt)) {
        showError(`File type not allowed. Allowed: ${allowedExts.join(', ')}`);
        return;
    }
    
    if (file.size > 5 * 1024 * 1024 * 1024) {
        showError('File exceeds 5GB limit');
        return;
    }
    
    // Show upload progress
    const uploadContent = document.getElementById('upload-content');
    const uploadProgress = document.getElementById('upload-progress');
    uploadContent.style.display = 'none';
    uploadProgress.style.display = 'block';
    
    try {
        const result = await uploadFile(file);
        uploadedFileJobId = result.job_id;
        uploadedFilename.textContent = result.filename;
        
        uploadProgress.style.display = 'none';
        uploadContent.style.display = 'block';
        uploadSuccess.style.display = 'block';
        setupSection.style.display = 'block';
    } catch (error) {
        showError(`Upload failed: ${error.message}`);
        uploadProgress.style.display = 'none';
        uploadContent.style.display = 'block';
    }
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                const progressFill = document.getElementById('upload-progress-fill');
                const progressText = document.getElementById('upload-progress-text');
                const uploadStatus = document.querySelector('.upload-status');
                
                progressFill.style.width = `${percentComplete}%`;
                progressText.textContent = `${Math.round(percentComplete)}%`;
                uploadStatus.textContent = `Uploading ${file.name}...`;
            }
        });
        
        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const data = JSON.parse(xhr.responseText);
                    resolve(data);
                } catch (error) {
                    reject(new Error('Failed to parse response'));
                }
            } else {
                try {
                    const error = JSON.parse(xhr.responseText);
                    reject(new Error(error.error || 'Upload failed'));
                } catch {
                    reject(new Error(`Upload failed: ${xhr.statusText}`));
                }
            }
        });
        
        xhr.addEventListener('error', () => {
            reject(new Error('Upload failed: Network error'));
        });
        
        xhr.open('POST', '/api/upload');
        xhr.send(formData);
    });
}

async function loadProjects() {
    try {
        const response = await fetch('/api/projects');
        const data = await response.json();
        
        projectSelect.innerHTML = '<option value="">-- Select a project or create new --</option>';
        
        if (data.projects && data.projects.length > 0) {
            // Store projects data
            projectSelect.dataset.projects = JSON.stringify(data.projects);
            
            data.projects.forEach(project => {
                const option = document.createElement('option');
                option.value = project.project_dir;
                option.textContent = `${project.name} (${project.frame_count} frames)`;
                projectSelect.appendChild(option);
            });
            
            loadProjectBtn.style.display = 'inline-block';
        }
    } catch (error) {
        console.error('Failed to load projects:', error);
    }
}

// Setup
document.getElementById('start-setup-btn').addEventListener('click', async () => {
    if (!uploadedFileJobId) {
        showError('Please upload a video file first');
        return;
    }
    
    const maxFrames = parseInt(document.getElementById('max-frames').value) || 500;
    const saveFrames = parseInt(document.getElementById('save-frames').value) || 10;
    
    try {
        const response = await fetch('/api/setup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_id: uploadedFileJobId,
                max_frames: maxFrames,
                save_frames: saveFrames
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Setup failed');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        
        setupSection.style.display = 'none';
        progressSection.style.display = 'block';
        pollProgress(currentJobId, async () => {
            // Setup complete, show training section
            await loadProjects(); // Refresh project list
            
            // Get project directory from result
            try {
                const progressResponse = await fetch(`/api/progress/${currentJobId}`);
                const progressData = await progressResponse.json();
                if (progressData.result && progressData.result.project_dir) {
                    currentProjectDir = progressData.result.project_dir;
                }
            } catch (error) {
                console.error('Failed to get project directory:', error);
            }
            
            trainingSection.style.display = 'block';
            productionSection.style.display = 'block';
        });
    } catch (error) {
        showError(error.message);
    }
});

// Training
document.getElementById('start-training-btn').addEventListener('click', async () => {
    if (!currentProjectDir) {
        showError('Please select or create a project first');
        return;
    }
    
    const query = document.getElementById('detector-query').value.trim();
    const detectorId = document.getElementById('detector-id').value.trim();
    
    if (!query && !detectorId) {
        showError('Please provide either a query or detector ID');
        return;
    }
    
    const config = {
        project_dir: currentProjectDir,
        query: query || null,
        detector_id: detectorId || null,
        confidence: parseFloat(document.getElementById('confidence').value) || 0.75,
        num_frames: parseInt(document.getElementById('num-frames').value) || 100,
        wait: parseFloat(document.getElementById('wait').value) || 120.0,
        ask_async: document.getElementById('ask-async').checked,
        human_review: document.getElementById('human-review').value
    };
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Training failed');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        
        trainingSection.style.display = 'none';
        progressSection.style.display = 'block';
        pollProgress(currentJobId, () => {
            // Training complete
            trainingSection.style.display = 'block';
            productionSection.style.display = 'block';
        });
    } catch (error) {
        showError(error.message);
    }
});

// Production
document.getElementById('start-production-btn').addEventListener('click', async () => {
    if (!currentProjectDir) {
        showError('Please select or create a project first');
        return;
    }
    
    const detectorIdsStr = document.getElementById('detector-ids').value.trim();
    if (!detectorIdsStr) {
        showError('Please provide at least one detector ID');
        return;
    }
    
    const detectorIds = detectorIdsStr.split(/\s+/).filter(id => id.length > 0);
    
    const config = {
        project_dir: currentProjectDir,
        detector_ids: detectorIds,
        frame_stride: parseInt(document.getElementById('frame-stride').value) || 1,
        human_review: document.getElementById('human-review-prod').value
    };
    
    try {
        const response = await fetch('/api/produce', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Production failed');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        
        productionSection.style.display = 'none';
        progressSection.style.display = 'block';
        pollProgress(currentJobId, () => {
            // Production complete, show results
            showResults(currentJobId);
        });
    } catch (error) {
        showError(error.message);
    }
});

// Progress polling
function pollProgress(jobId, onComplete) {
    if (progressInterval) {
        clearInterval(progressInterval);
    }
    
    progressInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/progress/${jobId}`);
            const data = await response.json();
            
            updateProgress(data);
            
            if (data.status === 'completed') {
                clearInterval(progressInterval);
                progressInterval = null;
                if (onComplete) {
                    onComplete();
                }
            } else if (data.status === 'error') {
                clearInterval(progressInterval);
                progressInterval = null;
                showError(data.error || 'Job failed');
            }
        } catch (error) {
            console.error('Failed to poll progress:', error);
        }
    }, 2000); // Poll every 2 seconds
}

function updateProgress(data) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const statusMessage = document.getElementById('status-message');
    
    const progress = data.progress || 0;
    progressFill.style.width = `${progress}%`;
    progressText.textContent = `${Math.round(progress)}%`;
    statusMessage.textContent = data.message || 'Processing...';
}

// Results
function showResults(jobId) {
    progressSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    const downloadBtn = document.getElementById('download-video');
    const previewBtn = document.getElementById('preview-video');
    
    // Set preview button (no retry needed, server handles it)
    previewBtn.href = `/api/preview/${jobId}/video`;
    previewBtn.style.display = 'inline-block';
    
    // Set up download button with retry logic
    downloadBtn.style.display = 'inline-block';
    downloadBtn.onclick = async (e) => {
        e.preventDefault();
        await downloadVideoWithRetry(jobId);
    };
}

// Download video with retry logic
async function downloadVideoWithRetry(jobId, maxRetries = 3) {
    const downloadBtn = document.getElementById('download-video');
    const originalText = downloadBtn.textContent;
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            downloadBtn.textContent = attempt === 0 
                ? 'Downloading...' 
                : `Retrying download (${attempt + 1}/${maxRetries})...`;
            downloadBtn.disabled = true;
            
            // Add small delay before first attempt to allow filesystem sync
            if (attempt === 0) {
                await new Promise(resolve => setTimeout(resolve, 300));
            }
            
            const response = await fetch(`/api/download/${jobId}/video`, {
                method: 'GET',
            });
            
            if (response.ok) {
                // Get the blob and trigger download
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                
                // Get filename from Content-Disposition header if available
                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = `video-${jobId}.mp4`;
                if (contentDisposition) {
                    const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
                    if (filenameMatch && filenameMatch[1]) {
                        filename = filenameMatch[1].replace(/['"]/g, '');
                    }
                }
                
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                downloadBtn.textContent = originalText;
                downloadBtn.disabled = false;
                return; // Success
            } else {
                const error = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(error.error || `HTTP ${response.status}`);
            }
        } catch (error) {
            console.error(`Download attempt ${attempt + 1} failed:`, error);
            
            if (attempt < maxRetries - 1) {
                // Exponential backoff: 500ms, 1000ms, 2000ms
                const delay = 500 * Math.pow(2, attempt);
                await new Promise(resolve => setTimeout(resolve, delay));
            } else {
                // All retries failed
                downloadBtn.textContent = originalText;
                downloadBtn.disabled = false;
                showError(`Failed to download video after ${maxRetries} attempts: ${error.message}`);
            }
        }
    }
}

// New job
document.getElementById('new-job').addEventListener('click', () => {
    // Reset state
    currentJobId = null;
    currentProjectDir = null;
    uploadedFileJobId = null;
    videoFileInput.value = '';
    
    // Hide all sections
    uploadSection.style.display = 'none';
    setupSection.style.display = 'none';
    trainingSection.style.display = 'none';
    productionSection.style.display = 'none';
    progressSection.style.display = 'none';
    resultsSection.style.display = 'none';
    uploadSuccess.style.display = 'none';
    
    // Reload projects
    loadProjects();
    
    hideError();
});

// Error handling
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'block';
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideError() {
    errorMessage.style.display = 'none';
}

