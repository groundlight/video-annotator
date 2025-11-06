"""
Video Preview Module - Compartmentalized video serving functionality.

This module handles all video preview and serving logic, allowing it to be
modified independently from the main webapp.
"""

import logging
import json
import time
from pathlib import Path
from flask import send_file, jsonify

logger = logging.getLogger(__name__)


def verify_file_accessible(filepath, max_checks=3, delay=0.1):
    """
    Verify that a file exists and is readable (not locked).
    
    Args:
        filepath: Path object to verify
        max_checks: Maximum number of verification attempts
        delay: Delay between attempts in seconds
        
    Returns:
        True if file is accessible, False otherwise
    """
    for attempt in range(max_checks):
        try:
            if filepath.exists() and filepath.stat().st_size > 0:
                # Try to open file to ensure it's not locked
                with open(filepath, 'rb') as f:
                    f.read(1)
                return True
        except (OSError, IOError, PermissionError) as e:
            logger.debug(f"File verification attempt {attempt + 1} failed: {e}")
            if attempt < max_checks - 1:
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            logger.debug(f"Unexpected error verifying file: {e}")
            if attempt < max_checks - 1:
                time.sleep(delay * (attempt + 1))
    
    return False


def find_video_path(job_id, jobs_dict, projects_folder):
    """
    Find video path from job or fallback to project directory scan.
    Includes retry logic and file verification for reliability.
    
    Args:
        job_id: Job ID to look up
        jobs_dict: Dictionary of jobs
        projects_folder: Path to projects folder
        
    Returns:
        Path object to video file, or None if not found
    """
    logger.debug(f"Finding video path for job_id={job_id}")
    
    # First, try to get from job
    if job_id in jobs_dict:
        job = jobs_dict[job_id]
        if job.get('status') == 'completed' and job.get('result'):
            video_path = job['result'].get('video_path')
            if video_path:
                filepath = Path(video_path)
                if not filepath.is_absolute():
                    filepath = filepath.resolve()
                
                # Verify file is accessible with retry
                if verify_file_accessible(filepath):
                    logger.debug(f"Found video path from job: {filepath}")
                    return filepath
                else:
                    logger.debug(f"Video path from job exists but not accessible: {filepath}")
            
            # Try to recover from project_dir if available
            project_dir = job['result'].get('project_dir')
            if project_dir:
                video_output_dir = Path(project_dir) / 'video_output'
                if video_output_dir.exists():
                    video_files = sorted(
                        video_output_dir.glob('*.mp4'),
                        key=lambda p: p.stat().st_mtime
                    )
                    if video_files:
                        video_path = video_files[-1].resolve()
                        # Verify file is accessible
                        if verify_file_accessible(video_path):
                            logger.debug(f"Recovered video path from project_dir: {video_path}")
                            return video_path
    
    # Fallback: scan project directories for most recent video
    logger.debug(f"Job {job_id} not found, scanning project directories")
    projects_path = Path(projects_folder)
    if not projects_path.exists():
        logger.warning(f"Projects folder does not exist: {projects_path}")
        return None
    
    project_dirs = [d for d in projects_path.iterdir() if d.is_dir()]
    most_recent_video = None
    most_recent_time = 0
    
    for project_dir in project_dirs:
        video_output_dir = project_dir / 'video_output'
        if video_output_dir.exists():
            video_files = sorted(
                video_output_dir.glob('*.mp4'),
                key=lambda p: p.stat().st_mtime
            )
            if video_files:
                latest_video = video_files[-1]
                mtime = latest_video.stat().st_mtime
                if mtime > most_recent_time:
                    most_recent_time = mtime
                    most_recent_video = latest_video.resolve()
    
    if most_recent_video:
        # Verify file is accessible
        if verify_file_accessible(most_recent_video):
            logger.debug(f"Found most recent video from project scan: {most_recent_video}")
            return most_recent_video
        else:
            logger.debug(f"Most recent video found but not accessible: {most_recent_video}")
    
    logger.warning(f"Could not find video path for job_id={job_id}")
    return None


def find_video_path_with_retry(job_id, jobs_dict, projects_folder, max_retries=3, delay=0.2):
    """
    Find video path with retry logic to handle filesystem sync delays.
    
    Args:
        job_id: Job ID to look up
        jobs_dict: Dictionary of jobs
        projects_folder: Path to projects folder
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        
    Returns:
        Path object to video file, or None if not found
    """
    for attempt in range(max_retries):
        filepath = find_video_path(job_id, jobs_dict, projects_folder)
        
        if filepath:
            return filepath
        
        if attempt < max_retries - 1:
            # Exponential backoff
            wait_time = delay * (attempt + 1)
            logger.debug(f"Retry {attempt + 1}/{max_retries} for job_id={job_id} after {wait_time}s")
            time.sleep(wait_time)
    
    return None


def serve_video_preview_page(job_id, video_path, video_url=None):
    """
    Generate HTML page with video player for preview.
    
    Args:
        job_id: Job ID (used for logging/identification)
        video_path: Path to video file (used for logging/identification)
        video_url: Optional custom URL to serve the video file. If None, constructs
                   standard URL from job_id: /api/preview/{job_id}/video/file
        
    Returns:
        HTML string with video player
    """
    if video_url is None:
        video_url = f'/api/preview/{job_id}/video/file'
    
    # Use regular string formatting to avoid conflicts with JavaScript template literals
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>Video Preview - Video Annotator</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .container {
            max-width: 1200px;
            width: 100%;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 20px;
        }
        .video-container {
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            position: relative;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        video {
            width: 100%;
            height: auto;
            display: block;
        }
        .loading {
            color: white;
            text-align: center;
            padding: 40px;
        }
        .loading-spinner {
            border: 4px solid #333;
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .progress-container {
            width: 100%;
            max-width: 500px;
            margin: 20px auto;
            background: #2a2a2a;
            border-radius: 4px;
            padding: 10px;
            display: none;
        }
        .progress-bar {
            width: 0%;
            height: 20px;
            background: #4CAF50;
            border-radius: 2px;
            transition: width 0.3s ease;
        }
        .progress-text {
            color: white;
            text-align: center;
            margin-top: 10px;
            font-size: 14px;
        }
        .error {
            color: #ff6b6b;
            text-align: center;
            padding: 20px;
            background: #2a2a2a;
            border-radius: 8px;
            margin-top: 20px;
        }
        .retry-button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 15px;
            transition: background 0.3s;
        }
        .retry-button:hover {
            background: #45a049;
        }
        .retry-button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        .status {
            color: #aaa;
            text-align: center;
            margin-top: 10px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Video Annotator - Annotated Video Preview</h1>
        <div class="video-container">
            <div id="loading" class="loading">
                <div class="loading-spinner"></div>
                <div>Loading video preview...</div>
            </div>
            <video id="videoPlayer" controls style="display: none;" preload="auto">
                <!-- Video source will be set by JavaScript -->
            </video>
        </div>
        <div id="progressContainer" class="progress-container">
            <div id="progressBar" class="progress-bar"></div>
            <div id="progressText" class="progress-text">0%</div>
        </div>
        <div id="status" class="status"></div>
        <div id="error" class="error" style="display: none;"></div>
    </div>
    <script>
        const video = document.getElementById('videoPlayer');
        const loadingDiv = document.getElementById('loading');
        const errorDiv = document.getElementById('error');
        const progressContainer = document.getElementById('progressContainer');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const statusDiv = document.getElementById('status');
        
        const videoUrl = ''' + json.dumps(video_url) + ''';
        let retryCount = 0;
        const maxRetries = 3;
        let retryTimeout = null;
        
        function updateStatus(message) {
            statusDiv.textContent = message;
            console.log('Status:', message);
        }
        
        function showError(message, details, showRetry) {
            errorDiv.innerHTML = '<strong>' + message + '</strong>';
            if (details && details.length > 0) {
                errorDiv.innerHTML += '<br><small style="margin-top: 10px; display: block;">' + details.join('<br>') + '</small>';
            }
            if (showRetry && retryCount < maxRetries) {
                errorDiv.innerHTML += '<button class="retry-button" onclick="retryLoad()">Retry (' + (maxRetries - retryCount) + ' attempts left)</button>';
            }
            errorDiv.style.display = 'block';
            loadingDiv.style.display = 'none';
            video.style.display = 'none';
            progressContainer.style.display = 'none';
        }
        
        function showProgress(percent) {
            progressContainer.style.display = 'block';
            progressBar.style.width = percent + '%';
            progressText.textContent = percent + '%';
        }
        
        function hideLoading() {
            loadingDiv.style.display = 'none';
            video.style.display = 'block';
            progressContainer.style.display = 'none';
        }
        
        function retryLoad() {
            if (retryTimeout) {
                clearTimeout(retryTimeout);
            }
            retryCount++;
            updateStatus('Retrying... (Attempt ' + retryCount + ' of ' + maxRetries + ')');
            errorDiv.style.display = 'none';
            loadingDiv.style.display = 'block';
            video.style.display = 'none';
            // Use retry function that checks HTTP status first
            loadVideoWithRetry();
        }
        
        // Load video with HTTP status check and retry logic
        async function loadVideoWithRetry() {
            // First check if the endpoint is accessible
            try {
                updateStatus('Checking video availability...');
                const response = await fetch(videoUrl, { method: 'HEAD' });
                if (!response.ok) {
                    if (response.status === 404) {
                        // 404 error - retry with delay
                        if (retryCount < maxRetries) {
                            updateStatus('Video not found. Retrying in 2 seconds... (Attempt ' + (retryCount + 1) + ' of ' + maxRetries + ')');
                            retryTimeout = setTimeout(function() {
                                retryLoad();
                            }, 2000);
                            return;
                        } else {
                            showError('Video not found after retries', ['HTTP 404', 'URL: ' + videoUrl], true);
                            return;
                        }
                    } else {
                        // Other HTTP error
                        if (retryCount < maxRetries) {
                            updateStatus('Server error (HTTP ' + response.status + '). Retrying in 2 seconds...');
                            retryTimeout = setTimeout(function() {
                                retryLoad();
                            }, 2000);
                            return;
                        }
                    }
                }
            } catch (networkError) {
                // Network error checking endpoint
                if (retryCount < maxRetries) {
                    updateStatus('Cannot reach video server. Retrying in 2 seconds... (Attempt ' + (retryCount + 1) + ' of ' + maxRetries + ')');
                    retryTimeout = setTimeout(function() {
                        retryLoad();
                    }, 2000);
                    return;
                } else {
                    showError('Cannot reach video server after retries', ['Network error', 'URL: ' + videoUrl], true);
                    return;
                }
            }
            
            // If we get here, endpoint is accessible, proceed with normal video load
            loadVideo();
        }
        
        function loadVideo() {
            updateStatus('Connecting to video server...');
            loadingDiv.style.display = 'block';
            video.style.display = 'none';
            errorDiv.style.display = 'none';
            
            // Set video source
            video.src = videoUrl;
            video.load();
        }
        
        // Track loading progress
        let lastProgress = 0;
        video.addEventListener('progress', function() {
            if (video.buffered.length > 0 && video.duration > 0) {
                const bufferedEnd = video.buffered.end(video.buffered.length - 1);
                const percent = Math.round((bufferedEnd / video.duration) * 100);
                if (percent !== lastProgress) {
                    lastProgress = percent;
                    showProgress(percent);
                    updateStatus('Loading video: ' + percent + '%');
                }
            }
        });
        
        // Video loading started
        video.addEventListener('loadstart', function() {
            updateStatus('Starting video load...');
            errorDiv.style.display = 'none';
        });
        
        // Metadata loaded
        video.addEventListener('loadedmetadata', function() {
            updateStatus('Video metadata loaded. Duration: ' + Math.round(video.duration) + 's');
            showProgress(10);
        });
        
        // Data loaded
        video.addEventListener('loadeddata', function() {
            updateStatus('Video data loaded');
            showProgress(30);
        });
        
        // Can start playing
        video.addEventListener('canplay', function() {
            updateStatus('Video ready to play');
            showProgress(100);
            setTimeout(function() {
                hideLoading();
                updateStatus('Video loaded successfully');
            }, 500);
        });
        
        // Can play through
        video.addEventListener('canplaythrough', function() {
            updateStatus('Video fully loaded');
            showProgress(100);
        });
        
        // Playing
        video.addEventListener('playing', function() {
            updateStatus('Video playing');
        });
        
        // Error handling with auto-retry on multiple error types
        video.addEventListener('error', function(e) {
            let errorMsg = 'Failed to load video. ';
            let errorDetails = [];
            
            if (video.error) {
                errorDetails.push('Error code: ' + video.error.code);
                
                switch(video.error.code) {
                    case video.error.MEDIA_ERR_ABORTED:
                        errorMsg += 'Video playback aborted.';
                        break;
                    case video.error.MEDIA_ERR_NETWORK:
                        errorMsg += 'Network error while loading video.';
                        errorDetails.push('The server may be temporarily unavailable.');
                        break;
                    case video.error.MEDIA_ERR_DECODE:
                        errorMsg += 'Video decoding error.';
                        errorDetails.push('The video file may be corrupted or in an unsupported format.');
                        break;
                    case video.error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                        errorMsg += 'Video format not supported.';
                        errorDetails.push('Your browser may not support this video codec.');
                        break;
                    default:
                        errorMsg += 'Unknown error (code: ' + video.error.code + ').';
                }
            }
            
            errorDetails.push('ReadyState: ' + video.readyState);
            errorDetails.push('NetworkState: ' + video.networkState);
            errorDetails.push('URL: ' + videoUrl);
            
            console.error('Video error:', errorMsg, errorDetails);
            
            // Auto-retry for network errors, decode errors (might be transient), format errors, and readyState 0
            // These errors might be due to filesystem sync delays or temporary issues
            const shouldRetry = (
                (video.error && (
                    video.error.code === video.error.MEDIA_ERR_NETWORK ||
                    video.error.code === video.error.MEDIA_ERR_DECODE ||
                    video.error.code === video.error.MEDIA_ERR_SRC_NOT_SUPPORTED
                )) ||
                video.readyState === 0
            ) && retryCount < maxRetries;
            
            if (shouldRetry) {
                updateStatus('Error detected. Retrying in 2 seconds... (Attempt ' + (retryCount + 1) + ' of ' + maxRetries + ')');
                retryTimeout = setTimeout(function() {
                    retryLoad();
                }, 2000);
            } else {
                showError(errorMsg, errorDetails, true);
            }
        });
        
        // Network issues
        video.addEventListener('stalled', function() {
            updateStatus('Video loading stalled - waiting for data...');
        });
        
        video.addEventListener('waiting', function() {
            updateStatus('Video buffering...');
        });
        
        // Abort handling
        video.addEventListener('abort', function() {
            updateStatus('Video load aborted');
        });
        
        // Load video when page is ready
        window.addEventListener('load', function() {
            console.log('Page loaded, starting video load...');
            updateStatus('Page loaded. Starting video load...');
            // Small delay to ensure page is fully rendered, then use retry function
            setTimeout(function() {
                loadVideoWithRetry();
            }, 100);
        });
    </script>
</body>
</html>'''
    return html


def serve_video_file(job_id, video_path, request, jobs_dict, projects_folder):
    """
    Serve video file with proper range request support and retry logic.
    
    Args:
        job_id: Job ID
        video_path: Path to video file (may be None)
        request: Flask request object
        jobs_dict: Dictionary of jobs
        projects_folder: Path to projects folder
        
    Returns:
        Flask response object
    """
    import time
    
    logger.debug(f"Serving video file for job_id={job_id}")
    
    # Resolve video path with retry logic
    filepath = None
    
    # First try with provided video_path if available
    if video_path:
        filepath = Path(video_path)
        if not filepath.is_absolute():
            filepath = filepath.resolve()
        # Verify file is accessible
        if not verify_file_accessible(filepath):
            filepath = None  # Fall back to finding path
    
    # Use retry logic to find video path
    if not filepath:
        filepath = find_video_path_with_retry(job_id, jobs_dict, projects_folder, max_retries=3, delay=0.2)
        if not filepath:
            logger.error(f"Could not find video path for job_id={job_id} after retries")
            return jsonify({'error': 'Video not found'}), 404
    
    # Final verification before serving
    if not filepath.exists():
        logger.error(f"Video file does not exist: {filepath}")
        return jsonify({'error': f'Video file not found: {filepath}'}), 404
    
    if not filepath.is_file():
        logger.error(f"Path is not a file: {filepath}")
        return jsonify({'error': 'Path is not a file'}), 400
    
    logger.info(f"Serving video file: {filepath}")
    
    # Use Flask's send_file with conditional=True for automatic range request support
    try:
        response = send_file(
            str(filepath),
            mimetype='video/mp4',
            as_attachment=False,  # Don't force download, allow browser to play
            conditional=True  # Support range requests for seeking
        )
        # Add headers for better browser video support (matching timeliner)
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Type'] = 'video/mp4'
        response.headers['Cache-Control'] = 'no-cache'
        return response
    except Exception as e:
        logger.error(f"Error serving video file {filepath}: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error serving video: {str(e)}'}), 500

