#!/usr/bin/env python3
"""
Flask web application for Video Annotator
"""

import os
import sys
import threading
import time
import uuid
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from logging.handlers import RotatingFileHandler

# Add src directory to Python path so modules can find each other
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import video annotator modules (now using the path added above)
from projstate import ProjectState
from framemgr import FrameManager
from s1setup import save_frames
from s3produce import infer_and_produce_video
from groundlight import Groundlight

# Import video preview module
from video_preview import find_video_path, find_video_path_with_retry, serve_video_preview_page, serve_video_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add file handler with rotation
file_handler = RotatingFileHandler(
    'webapp.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Set debug level if FLASK_DEBUG is set
if os.environ.get('FLASK_DEBUG') == '1':
    logging.getLogger().setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path('uploads')
PROJECTS_FOLDER = Path('proj')
TEST_VIDEOS_FOLDER = Path('test_videos')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
PROJECTS_FOLDER.mkdir(exist_ok=True)
TEST_VIDEOS_FOLDER.mkdir(exist_ok=True)

# In-memory storage for job progress
jobs = {}

# Maximum file size (5GB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['PROJECTS_FOLDER'] = str(PROJECTS_FOLDER)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def project_dir_from_filename(filename: str) -> str:
    """Get the project directory from the filename."""
    base_filename = os.path.basename(filename)
    base_filename = os.path.splitext(base_filename)[0]  # Remove the extension
    out = os.path.join(PROJECTS_FOLDER, base_filename)
    os.makedirs(out, exist_ok=True)
    return out


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    logger.info(f"Upload request received")
    
    if 'file' not in request.files:
        logger.warning("No file in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        logger.warning(f"File type not allowed: {file.filename}")
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    unique_filename = f"{timestamp}_{filename}"
    filepath = UPLOAD_FOLDER / unique_filename
    file.save(str(filepath))
    
    logger.info(f"File uploaded: {filepath}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'uploaded',
        'filepath': str(filepath),
        'filename': filename,
        'progress': 0,
        'message': f'File uploaded: {filename}',
        'result': None
    }
    
    return jsonify({
        'job_id': job_id,
        'filename': filename
    })


@app.route('/api/projects', methods=['GET'])
def list_projects():
    """List existing projects"""
    logger.info("Listing projects")
    
    projects = []
    if PROJECTS_FOLDER.exists():
        for project_dir in PROJECTS_FOLDER.iterdir():
            if project_dir.is_dir():
                project_info_path = project_dir / 'project-info.json'
                if project_info_path.exists():
                    try:
                        project = ProjectState.load(str(project_dir))
                        projects.append({
                            'name': project_dir.name,
                            'project_dir': str(project_dir),
                            'video_path': project.video_path,
                            'frame_count': len(project.frame_metadata),
                            'has_video_output': (project_dir / 'video_output').exists()
                        })
                    except Exception as e:
                        logger.warning(f"Error loading project {project_dir}: {e}")
                        continue
    
    logger.info(f"Found {len(projects)} projects")
    return jsonify({'projects': projects})


@app.route('/api/setup', methods=['POST'])
def start_setup():
    """Start project setup (s1setup.py equivalent)"""
    data = request.json
    job_id = data.get('job_id')
    max_frames = data.get('max_frames', 500)
    save_frames_count = data.get('save_frames', 10)
    
    logger.info(f"Starting setup job {job_id}: max_frames={max_frames}, save_frames={save_frames_count}")
    
    if job_id not in jobs:
        logger.warning(f"Job {job_id} not found")
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] != 'uploaded':
        logger.warning(f"Job {job_id} status is {job['status']}, expected 'uploaded'")
        return jsonify({'error': f"Job status is '{job['status']}', not 'uploaded'"}), 400
    
    # Start setup in background thread
    thread = threading.Thread(target=setup_project_job, args=(job_id, max_frames, save_frames_count))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'job_id': job_id})


def setup_project_job(job_id, max_frames, save_frames_count):
    """Process project setup in background thread"""
    job = jobs[job_id]
    
    try:
        job['status'] = 'processing'
        job['message'] = 'Starting project setup...'
        job['progress'] = 10
        
        logger.info(f"Setup job {job_id} started")
        
        filepath = Path(job['filepath'])
        if not filepath.exists():
            raise FileNotFoundError(f"Video file not found: {filepath}")
        
        # Determine project directory
        project_dir = project_dir_from_filename(job['filename'])
        logger.info(f"Project directory: {project_dir}")
        
        job['progress'] = 20
        job['message'] = 'Loading video and analyzing frames...'
        
        # Create project state
        project = ProjectState(project_dir=str(project_dir), video_path=str(filepath))
        
        # Create frame manager and analyze
        decoder = FrameManager(video_path=str(filepath), max_frames=max_frames)
        
        job['progress'] = 40
        job['message'] = 'Analyzing and clustering frames...'
        
        decoder.analyze()
        
        job['progress'] = 60
        job['message'] = 'Saving frames...'
        
        # Save frames
        if save_frames_count > 0:
            save_frames(decoder, num_frames=save_frames_count, save_dir=project.subdir("sample_frames"))
        
        job['progress'] = 80
        job['message'] = 'Saving project state...'
        
        # Save project state
        project.save()
        
        job['status'] = 'completed'
        job['progress'] = 100
        job['message'] = f'Project setup complete! Analyzed {len(decoder)} frames.'
        job['result'] = {
            'project_dir': project_dir,
            'frame_count': len(decoder),
            'video_path': str(filepath)
        }
        
        logger.info(f"Setup job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Setup job {job_id} failed: {e}", exc_info=True)
        job['status'] = 'error'
        job['message'] = f'Error: {str(e)}'
        import traceback
        job['error_traceback'] = traceback.format_exc()


@app.route('/api/train', methods=['POST'])
def start_training():
    """Start detector training (s2train.py equivalent)"""
    data = request.json
    project_dir = data.get('project_dir')
    query = data.get('query')
    detector_id = data.get('detector_id')
    confidence = data.get('confidence', 0.75)
    num_frames = data.get('num_frames', 100)
    wait = data.get('wait', 120.0)
    ask_async = data.get('ask_async', False)
    human_review = data.get('human_review', 'DEFAULT')
    
    logger.info(f"Starting training: project_dir={project_dir}, query={query}, detector_id={detector_id}")
    
    if not project_dir:
        return jsonify({'error': 'project_dir is required'}), 400
    
    if not query and not detector_id:
        return jsonify({'error': 'Either query or detector_id is required'}), 400
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'processing',
        'project_dir': project_dir,
        'progress': 0,
        'message': 'Starting detector training...',
        'result': None
    }
    
    # Start training in background thread
    thread = threading.Thread(
        target=train_detector_job,
        args=(job_id, project_dir, query, detector_id, confidence, num_frames, wait, ask_async, human_review)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'job_id': job_id})


def train_detector_job(job_id, project_dir, query, detector_id, confidence, num_frames, wait, ask_async, human_review):
    """Process detector training in background thread"""
    job = jobs[job_id]
    
    try:
        logger.info(f"Training job {job_id} started")
        
        # Load project
        project = ProjectState.load(project_dir)
        decoder = FrameManager.for_project(project)
        
        # Validate project has frame metadata with diversity_rank
        if len(project.frame_metadata) == 0:
            raise ValueError("Project has not been set up yet. Please run setup first.")
        
        has_diversity_rank = False
        for fmd in project.frame_metadata.frame_metadata:
            if "diversity_rank" in fmd:
                has_diversity_rank = True
                break
        
        if not has_diversity_rank:
            raise ValueError("Project frame metadata is missing diversity_rank. Please re-run setup to cluster frames.")
        
        job['progress'] = 10
        job['message'] = 'Connecting to Groundlight...'
        
        # Connect to Groundlight
        gl = Groundlight()
        
        # Build or get detector
        if query:
            detector_name = query[:20]
            detector = gl.get_or_create_detector(name=detector_name, query=query, confidence_threshold=confidence)
            job['message'] = f'Created detector {detector.id}'
        else:
            detector = gl.get_detector(detector_id)
            if confidence is not None:
                gl.update_detector_confidence_threshold(detector, confidence)
            job['message'] = f'Using detector {detector.id}'
        
        detector_id_str = detector.id
        logger.info(f"Using detector: {detector_id_str}")
        
        job['progress'] = 20
        job['message'] = 'Checking previously submitted frames...'
        
        # Get number of previously submitted frames
        num_previously_submitted = project.get_num_previously_submitted_frames(detector_id_str)
        
        num_frames_to_submit = min(num_frames, len(decoder))
        job['message'] = f'Submitting {num_frames_to_submit} frames (previously submitted: {num_previously_submitted})...'
        
        # Import s2train functions and create wrappers that pass gl explicitly
        # s2train uses a global gl variable, so we need to replicate the logic
        def submit_to_model_with_gl(gl, detector, fmd, ask_async, wait, human_review):
            """Submit frame to model with explicit gl client"""
            from s2train import pprint_iq
            
            iq_metadata = {"frame_num": fmd["frame_num"]}
            url = f"https://dashboard.groundlight.ai/reef/review/queue/detector/{detector.id}"
            
            if human_review == "ALWAYS":
                message = f"Image submitted to cloud labeler. If you wish to review yourself, you can open this URL:\n\t{url}"
            elif human_review == "DEFAULT":
                message = f'Image submitted with default escalation behavior. Image will only escalate to cloud labeler if the ML result is not confident. If you wish to review yourself, you can open this URL:\n\t{url}'
            elif human_review == "NEVER":
                message = f"Open the following URL in a browser to review the image:\n\t{url}"
            else:
                raise ValueError(f'Unexpected value for human_review: {human_review}')
            
            logger.debug(f"Submitting frame {fmd['frame_num']} to model")
            
            t1 = time.time()
            if ask_async:
                iq = gl.ask_async(detector, fmd["pil_img"], human_review=human_review, metadata=iq_metadata)
                logger.debug(f'Submitted {iq.id} asynchronously to Groundlight')
            else:
                iq = gl.submit_image_query(detector, fmd["pil_img"], wait=0.0, human_review=human_review, metadata=iq_metadata)
                logger.debug(f'Finished submitting {iq.id}')
                
                confidence = 0.0 if iq.result.confidence is None else iq.result.confidence
                confidence_threshold = detector.confidence_threshold
                if human_review == "ALWAYS":
                    iq = gl.wait_for_confident_result(iq, confidence_threshold=1.0, timeout_sec=wait)
                elif human_review == "DEFAULT" and confidence < confidence_threshold:
                    iq = gl.wait_for_confident_result(iq, confidence_threshold=confidence_threshold, timeout_sec=wait)
            
            t2 = time.time()
            elapsed_time = t2 - t1
            logger.debug(f'Result returned in {elapsed_time:.2f} seconds')
        
        def submit_to_model_retry_with_gl(gl, detector, fmd, ask_async, wait, human_review):
            """Submit frame with retry logic"""
            delay = 5
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    submit_to_model_with_gl(gl, detector, fmd, ask_async=ask_async, wait=wait, human_review=human_review)
                    break
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    logger.warning(f"Error submitting frame {fmd['frame_num']}: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
        
        # Submit frames
        i = 0
        num_submitted = 0
        total_frames = min(num_frames_to_submit, len(decoder))
        
        while num_submitted < total_frames:
            try:
                frame_num = decoder.frame_num_by_rank(i)
                i += 1
            except IndexError:
                job['message'] = f'Reached end of available frames. Submitted {num_submitted} of {num_frames_to_submit} requested.'
                break
            
            # Check if already submitted
            if project.check_frame_submission(frame_num, detector_id_str):
                continue
            
            # Update progress
            progress_pct = 20 + (num_submitted / total_frames) * 70
            job['progress'] = min(progress_pct, 90)
            job['message'] = f'Submitting frame {frame_num} ({num_submitted + 1}/{total_frames})...'
            
            # Submit frame
            fmd = decoder.framedat_by_num(frame_num)
            try:
                submit_to_model_retry_with_gl(gl, detector, fmd, ask_async=ask_async, wait=wait, human_review=human_review)
            finally:
                project.log_frame_submission(frame_num, detector_id_str)
                num_submitted += 1
        
        job['status'] = 'completed'
        job['progress'] = 100
        job['message'] = f'Training complete! Submitted {num_submitted} frames to detector {detector_id_str}.'
        job['result'] = {
            'detector_id': detector_id_str,
            'frames_submitted': num_submitted,
            'project_dir': project_dir
        }
        
        logger.info(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
        job['status'] = 'error'
        job['message'] = f'Error: {str(e)}'
        import traceback
        job['error_traceback'] = traceback.format_exc()


@app.route('/api/produce', methods=['POST'])
def start_production():
    """Start video production (s3produce.py equivalent)"""
    data = request.json
    project_dir = data.get('project_dir')
    detector_ids = data.get('detector_ids', [])
    frame_stride = data.get('frame_stride', 1)
    human_review = data.get('human_review', 'NEVER')
    
    logger.info(f"Starting production: project_dir={project_dir}, detector_ids={detector_ids}, frame_stride={frame_stride}")
    
    if not project_dir:
        return jsonify({'error': 'project_dir is required'}), 400
    
    if not detector_ids or len(detector_ids) == 0:
        return jsonify({'error': 'detector_ids is required'}), 400
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'processing',
        'project_dir': project_dir,
        'progress': 0,
        'message': 'Starting video production...',
        'result': None
    }
    
    # Start production in background thread
    thread = threading.Thread(
        target=produce_video_job,
        args=(job_id, project_dir, detector_ids, frame_stride, human_review)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'job_id': job_id})


def produce_video_job(job_id, project_dir, detector_ids, frame_stride, human_review):
    """Process video production in background thread"""
    job = jobs[job_id]
    
    try:
        logger.info(f"Production job {job_id} started")
        
        job['progress'] = 10
        job['message'] = 'Loading project...'
        
        # Load project
        project = ProjectState.load(project_dir)
        
        job['progress'] = 20
        job['message'] = 'Starting video production...'
        
        # Call infer_and_produce_video (includes optimization)
        # Note: web_preview_port=0 disables the web preview server
        infer_and_produce_video(
            project=project,
            detector_ids=detector_ids,
            frame_stride=frame_stride,
            web_preview_port=0,  # Disable web preview server
            human_review=human_review
        )
        
        # Wait a moment for filesystem to sync after optimization
        import time
        time.sleep(0.5)
        
        # Find the output video file with retry logic
        video_output_dir = Path(project_dir) / 'video_output'
        if not video_output_dir.exists():
            raise FileNotFoundError("video_output directory not found")
        
        # Retry finding file (in case optimization just finished)
        output_video = None
        for attempt in range(5):
            video_files = sorted(
                video_output_dir.glob('*.mp4'),
                key=lambda p: p.stat().st_mtime
            )
            if video_files:
                candidate = video_files[-1].resolve()
                # Verify file is readable and stable (not being written)
                try:
                    if candidate.exists() and candidate.stat().st_size > 0:
                        # Try to open file to ensure it's not locked
                        with open(candidate, 'rb') as f:
                            f.read(1)
                        output_video = candidate
                        break
                except (OSError, IOError):
                    pass  # File might still be locked, retry
            
            time.sleep(0.2)  # Wait before retry
        
        if not output_video:
            raise FileNotFoundError("No video files found or file not accessible")
        
        # Double-check file exists before marking complete
        if not output_video.exists():
            raise FileNotFoundError(f"Video file disappeared: {output_video}")
        
        job['status'] = 'completed'
        job['progress'] = 100
        job['message'] = f'Video production complete!'
        job['result'] = {
            'video_path': str(output_video),
            'project_dir': project_dir,
            'detector_ids': detector_ids
        }
        
        logger.info(f"Production job {job_id} completed successfully: {output_video}")
        
    except Exception as e:
        logger.error(f"Production job {job_id} failed: {e}", exc_info=True)
        job['status'] = 'error'
        job['message'] = f'Error: {str(e)}'
        import traceback
        job['error_traceback'] = traceback.format_exc()


@app.route('/api/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    """Get processing progress"""
    logger.debug(f"Progress request for job_id={job_id}")
    
    if job_id not in jobs:
        logger.warning(f"Job {job_id} not found")
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'status': job['status'],
        'progress': job.get('progress', 0),
        'message': job.get('message'),
    }
    
    if job['status'] == 'completed' and job.get('result'):
        response['result'] = job['result']
    elif job['status'] == 'error':
        response['error'] = job.get('message')
        if 'error_traceback' in job:
            response['error_traceback'] = job['error_traceback']
    
    return jsonify(response)


@app.route('/api/download/<job_id>/video', methods=['GET'])
def download_video(job_id):
    """Download annotated video with retry logic for reliability
    
    Uses the same robust fallback logic as the preview endpoint to find videos
    even if the job is not in memory (e.g., after server restart).
    Includes retry logic to handle filesystem sync delays.
    """
    import time
    
    logger.info(f"Download request for job_id={job_id}")
    
    # Use find_video_path_with_retry for reliability
    filepath = find_video_path_with_retry(job_id, jobs, str(PROJECTS_FOLDER), max_retries=3, delay=0.2)
    
    if not filepath:
        logger.warning(f"Could not find video for job_id={job_id} after retries")
        return jsonify({'error': 'Video not found'}), 404
    
    # Final verification before serving
    if not filepath.exists():
        logger.error(f"Video file not found: {filepath}")
        return jsonify({'error': 'Video file not found'}), 404
    
    logger.info(f"Sending video file for download: {filepath}")
    return send_file(str(filepath), as_attachment=True, download_name=filepath.name)


@app.route('/api/preview/<job_id>/video', methods=['GET'])
def preview_video(job_id):
    """Preview rendered video - serves HTML page with video player
    
    Uses the same robust fallback logic as the file endpoint to find videos
    even if the job is not in memory (e.g., after server restart).
    Includes retry logic to handle filesystem sync delays.
    """
    import time
    
    logger.info(f"Preview request for job_id={job_id}")
    
    # Use find_video_path_with_retry for reliability
    # This will:
    # 1. Try to get from jobs dictionary if available
    # 2. Fall back to scanning project directories for most recent video
    # 3. Retry with exponential backoff if file not immediately accessible
    filepath = find_video_path_with_retry(job_id, jobs, str(PROJECTS_FOLDER), max_retries=3, delay=0.2)
    
    if not filepath:
        logger.warning(f"Could not find video for job_id={job_id} after retries")
        return jsonify({'error': 'Video not found. The job may not exist or video file may have been moved.'}), 404
    
    logger.info(f"Serving preview page for video: {filepath}")
    
    # Use video preview module to generate HTML
    # The video_url will be constructed automatically as /api/preview/{job_id}/video/file
    return serve_video_preview_page(job_id, str(filepath))


@app.route('/api/preview/<job_id>/video/file', methods=['GET'])
def preview_video_file(job_id):
    """Serve the actual video file for playback"""
    logger.info(f"Video file request for job_id={job_id}")
    
    # Get video path from job if available
    video_path = None
    if job_id in jobs:
        job = jobs[job_id]
        if job.get('status') == 'completed' and job.get('result'):
            video_path = job['result'].get('video_path')
    
    # Use video preview module to serve file
    return serve_video_file(job_id, video_path, request, jobs, str(PROJECTS_FOLDER))


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    active_jobs = sum(1 for job in jobs.values() if job.get('status') == 'processing')
    
    return jsonify({
        'status': 'healthy',
        'jobs_count': len(jobs),
        'active_jobs': active_jobs,
        'projects_folder': str(PROJECTS_FOLDER),
        'uploads_folder': str(UPLOAD_FOLDER)
    })


@app.route('/api/debug/jobs', methods=['GET'])
def debug_jobs():
    """Debug endpoint to list all jobs (development only)"""
    if os.environ.get('FLASK_DEBUG') != '1':
        return jsonify({'error': 'Debug mode not enabled'}), 403
    
    return jsonify({
        'jobs': {
            job_id: {
                'status': job.get('status'),
                'progress': job.get('progress'),
                'message': job.get('message')
            }
            for job_id, job in jobs.items()
        }
    })


@app.route('/api/debug/projects', methods=['GET'])
def debug_projects():
    """Debug endpoint to list all project directories (development only)"""
    if os.environ.get('FLASK_DEBUG') != '1':
        return jsonify({'error': 'Debug mode not enabled'}), 403
    
    projects = []
    if PROJECTS_FOLDER.exists():
        for project_dir in PROJECTS_FOLDER.iterdir():
            if project_dir.is_dir():
                projects.append({
                    'name': project_dir.name,
                    'path': str(project_dir),
                    'has_project_info': (project_dir / 'project-info.json').exists(),
                    'has_video_output': (project_dir / 'video_output').exists()
                })
    
    return jsonify({'projects': projects})


@app.route('/api/debug/preview/test-video', methods=['GET'])
def debug_preview_test_video():
    """Debug endpoint to preview test video directly (development only)
    
    Query parameters:
    - file: Optional filename in test_videos/ folder (default: testing-video-preview-functionality.mp4)
    - path: Optional absolute path to any video file
    """
    # Check both environment variable and Flask app debug mode
    if os.environ.get('FLASK_DEBUG') != '1' and not app.debug:
        return jsonify({'error': 'Debug mode not enabled. Set FLASK_DEBUG=1 or run with debug=True'}), 403
    
    # Check if a custom path is provided
    custom_path = request.args.get('path')
    if custom_path:
        test_video_path = Path(custom_path)
        if not test_video_path.is_absolute():
            test_video_path = Path.cwd() / custom_path
    else:
        # Use filename parameter or default
        filename = request.args.get('file', 'testing-video-preview-functionality.mp4')
        test_video_path = TEST_VIDEOS_FOLDER / filename
    
    # Resolve to absolute path
    test_video_path = test_video_path.resolve()
    
    if not test_video_path.exists():
        logger.warning(f"Test video not found: {test_video_path}")
        return jsonify({'error': f'Test video not found: {test_video_path}'}), 404
    
    logger.info(f"Debug preview request for test video: {test_video_path}")
    
    # Use a special job_id for test video
    test_job_id = 'test-video-preview'
    
    # Use the debug endpoint URL for serving the video file
    # Include query parameters if provided
    debug_video_url = '/api/debug/preview/test-video/file'
    query_params = []
    if request.args.get('file'):
        query_params.append(f"file={request.args.get('file')}")
    if custom_path:
        query_params.append(f"path={request.args.get('path')}")
    if query_params:
        debug_video_url += '?' + '&'.join(query_params)
    
    # Generate preview page using video_preview module with custom URL
    return serve_video_preview_page(test_job_id, str(test_video_path), video_url=debug_video_url)


@app.route('/api/debug/preview/test-video/file', methods=['GET'])
def debug_preview_test_video_file():
    """Debug endpoint to serve test video file directly (development only)
    
    Query parameters:
    - file: Optional filename in test_videos/ folder (default: testing-video-preview-functionality.mp4)
    - path: Optional absolute path to any video file
    """
    # Check both environment variable and Flask app debug mode
    if os.environ.get('FLASK_DEBUG') != '1' and not app.debug:
        return jsonify({'error': 'Debug mode not enabled. Set FLASK_DEBUG=1 or run with debug=True'}), 403
    
    # Check if a custom path is provided
    custom_path = request.args.get('path')
    if custom_path:
        test_video_path = Path(custom_path)
        if not test_video_path.is_absolute():
            test_video_path = Path.cwd() / custom_path
    else:
        # Use filename parameter or default
        filename = request.args.get('file', 'testing-video-preview-functionality.mp4')
        test_video_path = TEST_VIDEOS_FOLDER / filename
    
    # Resolve to absolute path
    test_video_path = test_video_path.resolve()
    
    if not test_video_path.exists():
        logger.warning(f"Test video not found: {test_video_path}")
        return jsonify({'error': f'Test video not found: {test_video_path}'}), 404
    
    logger.info(f"Debug video file request for test video: {test_video_path}")
    
    # Use video_preview module to serve file
    return serve_video_file('test-video-preview', str(test_video_path), request, jobs, str(PROJECTS_FOLDER))


if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    Path('templates').mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True)
    Path('static/css').mkdir(exist_ok=True)
    Path('static/js').mkdir(exist_ok=True)
    
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_DEBUG') == '1'
    
    logger.info(f"Starting webapp on port {port} (debug={debug})")
    app.run(debug=debug, host='0.0.0.0', port=port)

