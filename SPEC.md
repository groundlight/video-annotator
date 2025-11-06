# Video Annotator - Technical Specification

## Overview

Video Annotator is a web-based application that efficiently extracts diverse frames from videos and sends them to Groundlight detectors for annotation and training. It uses clustering to find a diverse set of images that will train Groundlight detectors to high confidence levels with minimal annotation effort. The application provides a complete workflow from video upload through frame analysis, detector training, and annotated video production.

## Core Features

### 1. Video Upload & Project Setup

- **Input**: Supports standard video formats (MP4, AVI, MOV, MKV, WebM)
- **File Upload**: Drag-and-drop or file browser interface
- **File Size Limit**: Maximum 5GB per file
- **Project Creation**: Automatically creates project directories based on video filename
- **Frame Analysis**: 
  - Extracts frames from video using OpenCV
  - Configurable maximum frames to analyze (default: 500)
  - Generates embeddings for each frame using CNN
  - Clusters frames using k-means to find diverse representative frames
- **Frame Saving**: Saves top N most diverse frames as sample images (default: 10)
- **Project State**: Saves project metadata (`project-info.json`) and frame metadata (`frame-info.json`)

### 2. Frame Analysis & Clustering

- **Diversity-Based Selection**: Uses k-means clustering on CNN embeddings to find diverse frames
- **Frame Metadata**: Stores frame numbers, embeddings, cluster assignments, and diversity rankings
- **Efficient Processing**: Analyzes frames in batches for performance
- **Progress Tracking**: Real-time progress updates during analysis
- **Project Persistence**: All project data saved to `proj/<project_name>/` directory

### 3. Detector Training

- **Detector Creation**: Create new binary detectors with custom queries
- **Existing Detector Support**: Use existing detector IDs
- **Frame Submission**: Submit diverse frames to Groundlight in diversity order
- **Training Configuration**:
  - Confidence threshold (default: 0.75)
  - Number of frames to submit (default: 100)
  - Wait time for confident answers (default: 120 seconds)
  - Human review mode (NEVER, ALWAYS, DEFAULT)
  - Asynchronous submission option
- **Progress Tracking**: Real-time updates on frame submission progress
- **Submission Logging**: Tracks which frames have been submitted to avoid duplicates
- **Groundlight Integration**: Uses official Groundlight Python SDK

### 4. Video Production

- **Inference**: Runs detector inference on video frames
- **Annotated Video Generation**: Creates new video with detector labels overlaid
- **Multi-Detector Support**: Can use multiple detectors simultaneously
- **Frame Stride**: Configurable frequency for label updates (default: 1 = every frame)
- **Video FPS Preservation**: Output video FPS matches input video FPS (frame stride only affects label update frequency)
- **H.264 Encoding**: Videos automatically encoded with H.264 codec for browser compatibility
- **Video Optimization**: Automatic optimization with ffmpeg (faststart + H.264 re-encoding if needed)
- **Human Review Mode**: Configurable escalation behavior for production
- **Output Location**: Videos saved to `proj/<project_name>/video_output/` with timestamp
- **Progress Tracking**: Real-time progress during video production

### 5. Video Preview & Download

- **Video Preview**: Open annotated videos in new browser tab with HTML5 video player
- **Loading States**: Visual loading indicators, progress bar, and status messages
- **Auto-Retry**: Automatic retry logic for network errors, 404 errors, and transient format/decode errors
- **HTTP Status Checking**: Pre-checks video endpoint availability before loading
- **Range Request Support**: Proper HTTP Range request handling for video seeking
- **Download**: Download annotated video files directly with client-side retry logic
- **Retry Logic**: Server-side retry with exponential backoff for file finding and verification
- **File Verification**: Verifies file accessibility (exists, readable, not locked) before serving
- **Job Recovery**: Handles server restarts by scanning project directories for videos
- **Robust Path Resolution**: Absolute path handling with fallback logic for reliable file serving

## Web App Interface

### API Endpoints

#### Authentication
- No authentication required (localhost deployment)

#### Project Management

**GET `/api/projects`**
- List all existing projects from `proj/` directory
- Returns project metadata (name, video path, frame count, has_video_output)

**POST `/api/upload`**
- Upload video file
- Saves to `uploads/` directory
- Returns `job_id` and `filename`

#### Processing

**POST `/api/setup`**
- Start project setup job
- Parameters: `job_id`, `max_frames`, `save_frames`
- Background job: Analyzes frames, clusters, saves project state

**POST `/api/train`**
- Start detector training job
- Parameters: `project_dir`, `query` (or `detector_id`), `confidence`, `num_frames`, `wait`, `ask_async`, `human_review`
- Background job: Creates/uses detector, submits frames for training

**POST `/api/produce`**
- Start video production job
- Parameters: `project_dir`, `detector_ids[]`, `frame_stride`, `human_review`
- Background job: Runs inference, generates annotated video

#### Progress & Results

**GET `/api/progress/<job_id>`**
- Get job progress status
- Returns: `status`, `progress`, `message`, `result` (if completed), `error` (if failed)

**GET `/api/download/<job_id>/video`**
- Download annotated video file with retry logic
- Uses `find_video_path_with_retry()` for reliable file finding
- Uses Flask's `send_file` for download
- Client-side retry with exponential backoff

**GET `/api/preview/<job_id>/video`**
- Preview page with HTML5 video player
- Loading states, progress bar, and status messages
- Auto-retry on errors
- Delegates to `video_preview` module

**GET `/api/preview/<job_id>/video/file`**
- Serve video file with range request support and retry logic
- Uses `find_video_path_with_retry()` for reliable file finding
- Verifies file accessibility before serving
- Delegates to `video_preview` module

#### Debug & Health

**GET `/api/health`**
- Health check endpoint
- Returns server status, jobs count, active jobs

**GET `/api/debug/jobs`** (development only)
- List all jobs in memory
- Requires `FLASK_DEBUG=1`

**GET `/api/debug/projects`** (development only)
- List all project directories
- Requires `FLASK_DEBUG=1`

**GET `/api/debug/preview/test-video`** (development only)
- Preview test video directly for debugging
- Query parameters: `file` (filename in test_videos/) or `path` (absolute path)
- Requires `FLASK_DEBUG=1`

**GET `/api/debug/preview/test-video/file`** (development only)
- Serve test video file for debugging
- Query parameters: `file` (filename in test_videos/) or `path` (absolute path)
- Requires `FLASK_DEBUG=1`

### UI Components

The interface is organized into two main stages with clear navigation between them.

#### Stage 1: Choose Video and Train Detector

1. **Upload Section** (Primary - shown first)
   - Drag-and-drop area
   - File browser input
   - Upload progress bar
   - Success message with filename display
   - Toggle link: "Or load an existing project"

2. **Project Selection Section** (Secondary - hidden by default)
   - Dropdown of existing projects
   - "Load Project" button for existing projects
   - Toggle link: "Or upload a new video"
   - Shown when user clicks toggle link from upload section

3. **Setup Section**
   - Max frames input (default: 500)
   - Save frames count input (default: 10)
   - Start setup button
   - Shown after successful video upload

4. **Training Section**
   - Detector query input (for new detectors)
   - Detector ID input (for existing detectors)
   - Confidence threshold input
   - Number of frames input
   - Wait time input
   - Human review mode dropdown
   - Asynchronous submission checkbox
   - Start training button
   - Disabled until project setup is complete

5. **Ready for Stage 2 Message**
   - Shown after training completes
   - "Proceed to Stage 2" button
   - "Stay in Stage 1" button (allows continuing work in Stage 1)

6. **Start New Project Button**
   - Located at the end of Stage 1
   - Resets all state and returns to upload section

#### Stage 2: Produce Annotated Video

1. **Production Section**
   - Project selector dropdown
   - Detector IDs input (space-separated)
   - Frame stride input
   - Human review mode dropdown
   - Start production button

2. **Results Section** (only shown in Stage 2)
   - Download video button (with retry logic)
   - Preview video button (opens in new tab)
   - "Produce Another Video" button (replaces production form)

#### Shared Components

1. **Progress Section**
   - Progress bar with percentage
   - Status message display
   - Shown during any background job (setup, training, production)

2. **Error Display**
   - Error message card with icon
   - Auto-scrolls to error on display

#### Navigation

- **Stage Navigation**: Manual navigation buttons in stage headers ("Go to Stage 2" / "Go to Stage 1")
- **Stage 1 → Stage 2**: Can proceed automatically after training or navigate manually
- **Stage 2 → Stage 1**: Manual navigation only
- **Results Isolation**: Results from video production only appear in Stage 2, never in Stage 1

## Technical Architecture

### Project Structure

```
video-annotator/
├── webapp.py                    # Main Flask application
├── video_preview.py             # Compartmentalized video preview module
├── run_webapp.sh                # Helper script to run webapp
├── optimize_video_for_web.sh    # Video optimization script
├── templates/
│   └── index.html              # Main UI page
├── static/
│   ├── css/
│   │   └── style.css           # Styling
│   └── js/
│       └── app.js              # Frontend JavaScript
├── uploads/                     # Uploaded video files
├── proj/                        # Project directories
│   └── <project_name>/
│       ├── project-info.json    # Project metadata
│       ├── frame-info.json      # Frame metadata
│       ├── sample_frames/       # Diverse sample frames
│       └── video_output/        # Annotated videos (H.264 encoded)
├── test_videos/                 # Test video files for debugging
├── webapp.log                   # Application logs
├── SPEC.md                      # This file
├── WEB_BROWSER_COMPATIBILITY.md # Browser compatibility guidelines
├── VIDEO_OPTIMIZATION.md        # Video optimization documentation
├── INSTALL_FFMPEG.md            # ffmpeg installation instructions
└── requirements.txt             # Python dependencies
```

### Video Preview Module Design

The `video_preview.py` module is designed to be completely compartmentalized, allowing video preview functionality to be modified independently from the main webapp.

**Key Functions:**
- `verify_file_accessible()`: Verifies file exists, has size > 0, and is readable (not locked)
- `find_video_path()`: Resolves video file path from job or project directory scan with file verification
- `find_video_path_with_retry()`: Finds video path with retry logic and exponential backoff
- `serve_video_preview_page()`: Generates HTML page with video player, loading states, and auto-retry
- `serve_video_file()`: Serves video file with proper range request support and retry logic

**Video Serving Strategy:**
- Uses Flask's `send_file()` with `conditional=True` for automatic HTTP Range request handling
- Sets proper headers: `Accept-Ranges: bytes`, `Content-Type: video/mp4`, `Cache-Control: no-cache`
- Retry logic with exponential backoff (3 retries, 0.2s base delay)
- File verification before serving (exists, readable, not locked)
- Handles job lookup failures (e.g., after server restart)
- Falls back to scanning project directories for video files
- Ensures absolute paths for reliable file serving

**Video Preview Features:**
- Loading spinner and progress bar
- HTTP status checking before video load
- Auto-retry on network, decode, format errors (up to 3 attempts)
- Clear status messages during all loading stages
- Error display with retry button
- Preload metadata for faster initial load

**Error Handling:**
- Job not found → attempts to recover from project directory
- Video file not found → retry with exponential backoff
- File read errors → retry logic, proper error responses with logging
- Network errors → automatic retry with status updates
- Format/decode errors → automatic retry (may be transient)

### Background Job Processing

- **Threading**: Long-running tasks run in background threads to keep UI responsive
- **Job Tracking**: In-memory `jobs` dictionary stores job status and progress
- **Progress Updates**: Jobs update progress percentage and message during processing
- **Error Handling**: Exceptions caught and stored in job with full traceback
- **State Management**: Jobs transition through states: `uploaded` → `processing` → `completed`/`error`

### Logging

- **Structured Logging**: Uses Python's `logging` module
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Rotation**: RotatingFileHandler with 10MB max size, 5 backup files
- **Log File**: `webapp.log` in project root
- **Console Output**: Logs also output to console
- **Debug Mode**: Additional DEBUG-level logging when `FLASK_DEBUG=1`

### Configuration

- **Port**: Default 5001, configurable via `PORT` environment variable
- **Debug Mode**: Enable via `FLASK_DEBUG=1` environment variable
- **Max File Size**: 5GB (configurable in Flask app config)
- **CORS**: Enabled for cross-origin requests

## Error Handling

- **File Upload Errors**: Validation for file type and size
- **Job Errors**: Full error tracebacks stored in job state
- **API Errors**: Proper HTTP status codes and error messages
- **Video Preview Errors**: Graceful fallback to project directory scan with retry logic
- **Video Download Errors**: Client-side and server-side retry with exponential backoff
- **File Access Errors**: Verification and retry logic for filesystem sync delays
- **Groundlight API Errors**: Retry logic with exponential backoff
- **UI Error Display**: Clear error messages shown to user with retry options
- **Network Errors**: Automatic retry on video loading failures
- **Format Errors**: Automatic retry on transient codec/format issues

## Performance Considerations

- **Frame Analysis**: Processes frames in batches for memory efficiency
- **Background Processing**: Long-running tasks don't block UI
- **Progress Polling**: Frontend polls every 2 seconds (configurable)
- **Video File Serving**: Efficient streaming with range request support
- **Video Optimization**: Automatic H.264 encoding and faststart optimization for web playback
- **Metadata Preloading**: Video player uses `preload="metadata"` for faster initial load
- **Retry Logic**: Exponential backoff prevents overwhelming the system
- **File Verification**: Efficient checks to avoid serving locked files
- **Log Rotation**: Prevents log files from growing unbounded

## Security Considerations

- **Localhost Only**: Designed for local deployment (binds to 0.0.0.0)
- **File Validation**: Validates file types and sizes
- **Path Sanitization**: Uses `secure_filename` for uploaded files
- **No Authentication**: Intended for single-user local use

## Browser Compatibility

All generated videos are optimized for web browser playback:

- **Video Codec**: H.264 (libx264) - Required for browser compatibility
- **Audio Codec**: AAC or copy original
- **Container**: MP4 with moov atom at beginning (faststart)
- **Optimization**: Automatic re-encoding with ffmpeg if needed
- **Format Support**: MP4 with H.264 is universally supported by modern browsers

See `WEB_BROWSER_COMPATIBILITY.md` for detailed guidelines.

## Video Optimization

Videos are automatically optimized for web streaming:

- **Faststart**: moov atom moved to beginning for progressive playback
- **H.264 Encoding**: Re-encoded to browser-compatible codec
- **Quality Settings**: CRF 23 (high quality, reasonable file size)
- **Requirement**: ffmpeg must be installed (see `INSTALL_FFMPEG.md`)

Manual optimization script available: `optimize_video_for_web.sh`

See `VIDEO_OPTIMIZATION.md` for details.

## Version History

### Version 1.2.0 (Current)
- **UI Reorganization**: Two-stage workflow with improved navigation
  - Stage 1: Upload section is primary, project selection is secondary with toggle
  - Stage 2: Results section only appears in Stage 2, never in Stage 1
  - "Start New Project" button moved to end of Stage 1
  - "Produce Another Video" button replaces "Start New Project" in Stage 2 results
  - Manual navigation between stages via header buttons
  - Results isolation: production results only shown in Stage 2

### Version 1.1.0
- Added H.264 encoding for browser compatibility
- Automatic video optimization with ffmpeg (faststart + H.264)
- Retry logic for download and preview endpoints
- File verification and accessibility checks
- Enhanced video preview with loading states and auto-retry
- Client-side retry for downloads
- Debug endpoints for test video preview
- Comprehensive documentation (WEB_BROWSER_COMPATIBILITY.md, VIDEO_OPTIMIZATION.md)
- Video preview improvements (progress bar, status messages, error handling)

### Version 1.0.0 (Initial Release)
- Web-based interface for video-annotator
- Video upload and project setup
- Detector training interface
- Annotated video production
- Video preview and download
- Project management (create new, load existing)
- Compartmentalized video preview module
- Structured logging and error handling

