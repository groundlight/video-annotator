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
- **Human Review Mode**: Configurable escalation behavior for production
- **Output Location**: Videos saved to `proj/<project_name>/video_output/` with timestamp
- **Progress Tracking**: Real-time progress during video production

### 5. Video Preview & Download

- **Video Preview**: Open annotated videos in new browser tab with HTML5 video player
- **Range Request Support**: Proper HTTP Range request handling for video seeking
- **Download**: Download annotated video files directly
- **Job Recovery**: Handles server restarts by scanning project directories for videos
- **Robust Path Resolution**: Absolute path handling for reliable file serving

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
- Download annotated video file
- Uses Flask's `send_file` for download

**GET `/api/preview/<job_id>/video`**
- Preview page with HTML5 video player
- Delegates to `video_preview` module

**GET `/api/preview/<job_id>/video/file`**
- Serve video file with range request support
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

### UI Components

1. **Project Selection**
   - Dropdown of existing projects
   - "Create New Project" button
   - "Load Project" button for existing projects

2. **Upload Section** (for new projects)
   - Drag-and-drop area
   - File browser input
   - Upload progress bar
   - Success message with filename display

3. **Setup Section**
   - Max frames input (default: 500)
   - Save frames count input (default: 10)
   - Start setup button

4. **Training Section**
   - Detector query input (for new detectors)
   - Detector ID input (for existing detectors)
   - Confidence threshold input
   - Number of frames input
   - Wait time input
   - Human review mode dropdown
   - Asynchronous submission checkbox
   - Start training button

5. **Production Section**
   - Detector IDs input (space-separated)
   - Frame stride input
   - Human review mode dropdown
   - Start production button

6. **Progress Section**
   - Progress bar with percentage
   - Status message display

7. **Results Section**
   - Download video button
   - Preview video button (opens in new tab)
   - Start new project button

8. **Error Display**
   - Error message card with icon
   - Auto-scrolls to error on display

## Technical Architecture

### Project Structure

```
video-annotator/
├── webapp.py                    # Main Flask application
├── video_preview.py             # Compartmentalized video preview module
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
│       └── video_output/        # Annotated videos
└── webapp.log                   # Application logs
```

### Video Preview Module Design

The `video_preview.py` module is designed to be completely compartmentalized, allowing video preview functionality to be modified independently from the main webapp.

**Key Functions:**
- `find_video_path()`: Resolves video file path from job or project directory scan
- `serve_video_preview_page()`: Generates HTML page with video player
- `serve_video_file()`: Serves video file with proper range request support

**Video Serving Strategy:**
- Uses Flask's `send_file()` with `conditional=True` for automatic HTTP Range request handling
- Sets proper headers: `Accept-Ranges: bytes`, `Content-Type: video/mp4`, `Cache-Control: no-cache`
- Handles job lookup failures (e.g., after server restart)
- Falls back to scanning project directories for video files
- Ensures absolute paths for reliable file serving

**Error Handling:**
- Job not found → attempts to recover from project directory
- Video file not found → clear error message
- File read errors → proper error responses with logging

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
- **Video Preview Errors**: Graceful fallback to project directory scan
- **Groundlight API Errors**: Retry logic with exponential backoff
- **UI Error Display**: Clear error messages shown to user

## Performance Considerations

- **Frame Analysis**: Processes frames in batches for memory efficiency
- **Background Processing**: Long-running tasks don't block UI
- **Progress Polling**: Frontend polls every 2 seconds (configurable)
- **Video File Serving**: Efficient streaming with range request support
- **Log Rotation**: Prevents log files from growing unbounded

## Security Considerations

- **Localhost Only**: Designed for local deployment (binds to 0.0.0.0)
- **File Validation**: Validates file types and sizes
- **Path Sanitization**: Uses `secure_filename` for uploaded files
- **No Authentication**: Intended for single-user local use

## Version History

### Version 1.0.0 (Initial Release)
- Web-based interface for video-annotator
- Video upload and project setup
- Detector training interface
- Annotated video production
- Video preview and download
- Project management (create new, load existing)
- Compartmentalized video preview module
- Structured logging and error handling

