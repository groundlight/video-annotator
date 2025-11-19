// Global state
let currentJobId = null;
let currentProjectDir = null;
let progressInterval = null;
let uploadedFileJobId = null;
let currentStage = 1; // 1, 2, or 3
let trainedDetectorId = null; // Store detector ID from Stage 2
let availableProjects = []; // Store list of available projects

// Stage completion tracking
let stageCompletion = {
    stage1: false,  // Project setup complete
    stage2: false,   // Detector trained
    stage3: false    // Video produced (optional)
};

// DOM elements
const stage1 = document.getElementById('stage-1');
const stage2 = document.getElementById('stage-2');
const stage3 = document.getElementById('stage-3');
const navStage1 = document.getElementById('nav-stage-1');
const navStage2 = document.getElementById('nav-stage-2');
const navStage3 = document.getElementById('nav-stage-3');
const navigationError = document.getElementById('navigation-error');
const navigationErrorText = document.getElementById('navigation-error-text');
const projectSelect = document.getElementById('project-select');
const projectSelectionSection = document.getElementById('project-selection-section');
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
const trainingDisabledMessage = document.getElementById('training-disabled-message');
const trainingContent = document.getElementById('training-content');
// CTA elements for stage completion
const stage1Cta = document.getElementById('stage-1-cta');
const stage2Cta = document.getElementById('stage-2-cta');
const proceedToStage2Cta = document.getElementById('proceed-to-stage-2-cta');
const proceedToStage3Cta = document.getElementById('proceed-to-stage-3-cta');
const productionSection = document.getElementById('production-section');
const productionProjectSelect = document.getElementById('production-project-select');
const progressSection = document.getElementById('progress-section');
const resultsSection = document.getElementById('results-section');
const errorMessage = document.getElementById('error-message');
const errorText = document.getElementById('error-text');
const newProjectStage1Btn = document.getElementById('new-project-stage1');
const produceAnotherBtn = document.getElementById('produce-another-btn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadProjects();
    showStage(1);
    // Show upload section by default, hide project selection
    uploadSection.style.display = 'block';
    projectSelectionSection.style.display = 'none';
    // Set up navigation button handlers
    setupNavigationHandlers();
    updateNavigationIndicators();
});

// Navigation Handlers Setup
function setupNavigationHandlers() {
    navStage1.addEventListener('click', () => showStage(1));
    navStage2.addEventListener('click', () => showStage(2));
    navStage3.addEventListener('click', () => showStage(3));
}

// Stage Management Functions
function showStage(stageNumber) {
    // Always allow navigation - validate and show error if needed, but still navigate
    const isValid = validateStageAccess(stageNumber);
    
    if (!isValid) {
        // Error message already shown by validateStageAccess
        // Still navigate to the stage so user can see what's needed
    } else {
        // Clear navigation error if access is valid
        clearNavigationError();
    }
    
    // Hide all stages
    stage1.style.display = 'none';
    stage2.style.display = 'none';
    stage3.style.display = 'none';
    
    // Show selected stage
    currentStage = stageNumber;
    if (stageNumber === 1) {
        stage1.style.display = 'block';
        // Show/hide Stage 1 CTA based on completion
        if (stageCompletion.stage1) {
            stage1Cta.style.display = 'block';
        } else {
            stage1Cta.style.display = 'none';
        }
    } else if (stageNumber === 2) {
        stage2.style.display = 'block';
        // Enable/disable training based on stage 1 completion
        if (stageCompletion.stage1) {
            enableTraining();
        } else {
            disableTraining();
        }
        // Show/hide Stage 2 CTA based on completion
        if (stageCompletion.stage2) {
            stage2Cta.style.display = 'block';
        } else {
            stage2Cta.style.display = 'none';
        }
    } else if (stageNumber === 3) {
        stage3.style.display = 'block';
        updateProductionDefaults();
    }
    
    // Hide progress and results sections initially (they'll show when needed)
    progressSection.style.display = 'none';
    if (stageNumber !== 3) {
        resultsSection.style.display = 'none';
    }
    
    hideError();
    updateNavigationIndicators();
}

// Validate Stage Access
function validateStageAccess(stageNumber) {
    if (stageNumber === 1) {
        // Stage 1 is always accessible
        return true;
    } else if (stageNumber === 2) {
        // Stage 2 requires Stage 1 complete
        if (!stageCompletion.stage1) {
            showNavigationError('Please complete Project Setup (Stage 1) before accessing Train Detector (Stage 2). You need to upload/select a project and run the frame analysis setup.');
            return false;
        }
        return true;
    } else if (stageNumber === 3) {
        // Stage 3 requires either:
        // 1. Stage 2 complete (detector trained in this session), OR
        // 2. At least one project exists (user can use existing detector IDs)
        if (!stageCompletion.stage2 && availableProjects.length === 0) {
            showNavigationError('Please complete Train Detector (Stage 2) before accessing Produce Annotated Video (Stage 3). You need to train a detector first, or load an existing project that has trained detectors.');
            return false;
        }
        return true;
    }
    return true;
}

// Navigation Error Functions
function showNavigationError(message) {
    navigationErrorText.textContent = message;
    navigationError.style.display = 'flex';
    // Scroll to navigation error
    navigationError.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function clearNavigationError() {
    navigationError.style.display = 'none';
    navigationErrorText.textContent = '';
}

// Update Navigation Indicators
function updateNavigationIndicators() {
    // Remove active class from all buttons
    navStage1.classList.remove('active', 'completed', 'disabled');
    navStage2.classList.remove('active', 'completed', 'disabled');
    navStage3.classList.remove('active', 'completed', 'disabled');
    
    // Set active stage
    if (currentStage === 1) {
        navStage1.classList.add('active');
    } else if (currentStage === 2) {
        navStage2.classList.add('active');
    } else if (currentStage === 3) {
        navStage3.classList.add('active');
    }
    
    // Show checkmarks for completed stages
    if (stageCompletion.stage1) {
        navStage1.classList.add('completed');
        navStage1.querySelector('.stage-checkmark').style.display = 'inline';
    } else {
        navStage1.querySelector('.stage-checkmark').style.display = 'none';
    }
    
    if (stageCompletion.stage2) {
        navStage2.classList.add('completed');
        navStage2.querySelector('.stage-checkmark').style.display = 'inline';
    } else {
        navStage2.querySelector('.stage-checkmark').style.display = 'none';
    }
    
    if (stageCompletion.stage3) {
        navStage3.classList.add('completed');
        navStage3.querySelector('.stage-checkmark').style.display = 'inline';
    } else {
        navStage3.querySelector('.stage-checkmark').style.display = 'none';
    }
    
    // Add visual indicator (but don't disable) for stages that aren't ready
    // Users can still click and navigate, but will see an error message
    if (!stageCompletion.stage1) {
        navStage2.classList.add('disabled');
        navStage3.classList.add('disabled');
    } else if (!stageCompletion.stage2) {
        navStage3.classList.add('disabled');
    }
}

function enableTraining() {
    // Training section is in Stage 2, so only update if we're in Stage 2
    if (currentStage === 2) {
        trainingDisabledMessage.style.display = 'none';
        trainingContent.style.display = 'block';
        trainingSection.classList.remove('training-section-disabled');
    }
}

function disableTraining() {
    // Training section is in Stage 2, so only update if we're in Stage 2
    if (currentStage === 2) {
        trainingDisabledMessage.style.display = 'block';
        trainingContent.style.display = 'none';
        trainingSection.classList.add('training-section-disabled');
    }
}

function checkProjectReady(project) {
    return project && project.frame_count > 0;
}

function updateProductionDefaults() {
    // Populate production project selector
    const projects = JSON.parse(projectSelect.dataset.projects || '[]');
    productionProjectSelect.innerHTML = '<option value="">-- Select a project --</option>';
    
    projects.forEach(project => {
        const option = document.createElement('option');
        option.value = project.project_dir;
        option.textContent = `${project.name} (${project.frame_count} frames)`;
        if (project.project_dir === currentProjectDir) {
            option.selected = true;
        }
        productionProjectSelect.appendChild(option);
    });
    
    // Pre-fill detector ID if available from Stage 1 (but don't require it)
    if (trainedDetectorId) {
        document.getElementById('detector-ids').value = trainedDetectorId;
    } else {
        // Clear detector IDs if no trained detector from Stage 1
        document.getElementById('detector-ids').value = '';
    }
}

// Project selection
if (createNewProjectBtn) {
    createNewProjectBtn.addEventListener('click', () => {
        showStage(1);
        uploadSection.style.display = 'block';
        projectSelectionSection.style.display = 'none';
        setupSection.style.display = 'none';
        // Reset stage completion when starting new project
        stageCompletion.stage1 = false;
        stageCompletion.stage2 = false;
        stageCompletion.stage3 = false;
        updateNavigationIndicators();
        // Hide CTAs
        stage1Cta.style.display = 'none';
        stage2Cta.style.display = 'none';
        hideError();
    });
}

// Toggle between upload and project selection
const toggleProjectSelection = document.getElementById('toggle-project-selection');
const toggleUploadSection = document.getElementById('toggle-upload-section');

if (toggleProjectSelection) {
    toggleProjectSelection.addEventListener('click', (e) => {
        e.preventDefault();
        uploadSection.style.display = 'none';
        projectSelectionSection.style.display = 'block';
    });
}

if (toggleUploadSection) {
    toggleUploadSection.addEventListener('click', (e) => {
        e.preventDefault();
        uploadSection.style.display = 'block';
        projectSelectionSection.style.display = 'none';
    });
}

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
        
        // Enable training if project is set up
        if (checkProjectReady(project)) {
            enableTraining();
            // Mark stage 1 as complete if project is ready
            stageCompletion.stage1 = true;
            updateNavigationIndicators();
            // Show Stage 1 CTA if we're in Stage 1
            if (currentStage === 1) {
                stage1Cta.style.display = 'block';
            }
        } else {
            disableTraining();
            stageCompletion.stage1 = false;
            updateNavigationIndicators();
            // Hide Stage 1 CTA if project isn't ready
            if (currentStage === 1) {
                stage1Cta.style.display = 'none';
            }
            showError('Project has not been set up yet. Please create a new project and run setup first.');
        }
        stage2Cta.style.display = 'none';
    }
});

// CTA Button Handlers
if (proceedToStage2Cta) {
    proceedToStage2Cta.addEventListener('click', () => {
        showStage(2);
    });
}

if (proceedToStage3Cta) {
    proceedToStage3Cta.addEventListener('click', () => {
        showStage(3);
    });
}

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
    disableTraining();
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
        disableTraining(); // Training disabled until setup is complete
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
        
        // Store available projects for validation
        availableProjects = data.projects || [];
        
        projectSelect.innerHTML = '<option value="">-- Select a project or create new --</option>';
        
        if (data.projects && data.projects.length > 0) {
            // Store projects data
            projectSelect.dataset.projects = JSON.stringify(data.projects);
            
            data.projects.forEach(project => {
                const option = document.createElement('option');
                option.value = project.project_dir;
                option.textContent = `${project.name} (${project.frame_count} frames)`;
                if (project.project_dir === currentProjectDir) {
                    option.selected = true;
                }
                projectSelect.appendChild(option);
            });
            
            loadProjectBtn.style.display = 'inline-block';
            
            // Update production project selector if in Stage 3
            if (currentStage === 3) {
                updateProductionDefaults();
            }
        } else {
            // No projects available - clear the stored list
            availableProjects = [];
        }
    } catch (error) {
        console.error('Failed to load projects:', error);
        availableProjects = [];
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
            // Setup complete, enable training
            await loadProjects(); // Refresh project list
            
            // Mark Stage 1 as complete
            stageCompletion.stage1 = true;
            updateNavigationIndicators();
            // Show Stage 1 CTA if we're in Stage 1
            if (currentStage === 1) {
                stage1Cta.style.display = 'block';
            }
            
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
            
            enableTraining();
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
        pollProgress(currentJobId, async () => {
            // Training complete - get result and show ready for Stage 2 message
            try {
                const progressResponse = await fetch(`/api/progress/${currentJobId}`);
                const progressData = await progressResponse.json();
                if (progressData.result && progressData.result.detector_id) {
                    trainedDetectorId = progressData.result.detector_id;
                }
            } catch (error) {
                console.error('Failed to get training result:', error);
            }
            
            progressSection.style.display = 'none';
            trainingSection.style.display = 'none';
            
            // Mark Stage 2 as complete
            stageCompletion.stage2 = true;
            updateNavigationIndicators();
            // Show Stage 2 CTA if we're in Stage 2
            if (currentStage === 2) {
                stage2Cta.style.display = 'block';
            }
        });
    } catch (error) {
        showError(error.message);
    }
});

// Production
document.getElementById('start-production-btn').addEventListener('click', async () => {
    // Get project from production selector (Stage 2) or use current project (Stage 1)
    const selectedProjectDir = productionProjectSelect.value || currentProjectDir;
    
    if (!selectedProjectDir) {
        showError('Please select a project');
        return;
    }
    
    const detectorIdsStr = document.getElementById('detector-ids').value.trim();
    if (!detectorIdsStr) {
        showError('Please provide at least one detector ID');
        return;
    }
    
    const detectorIds = detectorIdsStr.split(/\s+/).filter(id => id.length > 0);
    
    const config = {
        project_dir: selectedProjectDir,
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
        console.log(`Production job started with job_id: ${currentJobId}`);
        
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
    
    let consecutive404s = 0;
    const max404s = 10; // Stop polling after 10 consecutive 404s (20 seconds)
    
    progressInterval = setInterval(async () => {
        try {
            const url = `/api/progress/${jobId}`;
            const response = await fetch(url);
            
            // Check if response is OK before parsing JSON
            if (!response.ok) {
                if (response.status === 404) {
                    consecutive404s++;
                    // Job might not be created yet, continue polling for a bit
                    if (consecutive404s >= max404s) {
                        clearInterval(progressInterval);
                        progressInterval = null;
                        console.error(`Job ${jobId} not found after ${max404s * 2} seconds. The job may have failed to start.`);
                        showError(`Job ${jobId} not found after ${max404s * 2} seconds. The job may have failed to start.`);
                    } else if (consecutive404s === 1) {
                        // Log first 404 for debugging
                        console.warn(`Job ${jobId} not found (404). Will retry up to ${max404s} times.`);
                    }
                    // Continue polling on 404 (job might not be created yet)
                    return;
                } else {
                    // Other HTTP errors - stop polling
                    clearInterval(progressInterval);
                    progressInterval = null;
                    const errorText = await response.text();
                    console.error(`Failed to fetch progress: HTTP ${response.status} - ${errorText}`);
                    showError(`Failed to fetch progress: HTTP ${response.status} - ${errorText}`);
                    return;
                }
            }
            
            // Reset 404 counter on successful response
            if (consecutive404s > 0) {
                console.log(`Job ${jobId} found after ${consecutive404s} retries.`);
                consecutive404s = 0;
            }
            
            const data = await response.json();
            console.log(`Progress update: status=${data.status}, progress=${data.progress}%, message=${data.message}`);
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
            // Only log non-404 errors to avoid console spam
            if (!error.message.includes('404') && !error.message.includes('Unexpected token')) {
                console.error('Failed to poll progress:', error);
            }
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
    // Only show results in Stage 3
    if (currentStage !== 3) {
        return;
    }
    
    // Mark Stage 3 as complete (optional, for tracking)
    stageCompletion.stage3 = true;
    updateNavigationIndicators();
    
    progressSection.style.display = 'none';
    resultsSection.style.display = 'block';
    productionSection.style.display = 'none'; // Hide production form when showing results
    
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
    const originalHTML = downloadBtn.innerHTML;
    const originalHref = downloadBtn.href;
    
    // Store original text (extract from innerHTML)
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = originalHTML;
    const originalText = tempDiv.textContent.trim();
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            // Update button text while preserving structure
            const loadingText = attempt === 0 
                ? 'Downloading...' 
                : `Retrying download (${attempt + 1}/${maxRetries})...`;
            
            // Disable by removing href and adding disabled class
            downloadBtn.href = '#';
            downloadBtn.classList.add('disabled');
            downloadBtn.style.pointerEvents = 'none';
            downloadBtn.style.opacity = '0.6';
            
            // Update text while preserving SVG
            const svg = downloadBtn.querySelector('svg');
            if (svg) {
                downloadBtn.innerHTML = svg.outerHTML + ' ' + loadingText;
            } else {
                downloadBtn.textContent = loadingText;
            }
            
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
                
                // Restore button
                downloadBtn.innerHTML = originalHTML;
                downloadBtn.href = originalHref;
                downloadBtn.classList.remove('disabled');
                downloadBtn.style.pointerEvents = '';
                downloadBtn.style.opacity = '';
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
                // All retries failed - restore button
                downloadBtn.innerHTML = originalHTML;
                downloadBtn.href = originalHref;
                downloadBtn.classList.remove('disabled');
                downloadBtn.style.pointerEvents = '';
                downloadBtn.style.opacity = '';
                showError(`Failed to download video after ${maxRetries} attempts: ${error.message}`);
            }
        }
    }
}

// New project button (Stage 1)
if (newProjectStage1Btn) {
    newProjectStage1Btn.addEventListener('click', () => {
        // Reset state
        currentJobId = null;
        currentProjectDir = null;
        uploadedFileJobId = null;
        trainedDetectorId = null;
        videoFileInput.value = '';
        
        // Reset to Stage 1
        showStage(1);
        
        // Show upload section, hide others
        uploadSection.style.display = 'block';
        projectSelectionSection.style.display = 'none';
        setupSection.style.display = 'none';
        trainingSection.style.display = 'none';
        stage2Cta.style.display = 'none';
        uploadSuccess.style.display = 'none';
        
        // Reload projects
        loadProjects();
        
        hideError();
    });
}

// Produce another video button (Stage 2)
if (produceAnotherBtn) {
    produceAnotherBtn.addEventListener('click', () => {
        // Hide results, show production section
        resultsSection.style.display = 'none';
        productionSection.style.display = 'block';
        
        // Reset production form but keep project selected
        document.getElementById('detector-ids').value = trainedDetectorId || '';
        document.getElementById('frame-stride').value = '1';
        document.getElementById('human-review-prod').value = 'NEVER';
        
        hideError();
    });
}

// Error handling
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'block';
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideError() {
    errorMessage.style.display = 'none';
}

