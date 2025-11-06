# Web Browser Compatibility Guidelines

## Important Reminder for All Web Apps

**When generating files (videos, images, audio, etc.) that will be viewed in web browsers, ensure they use browser-compatible formats and codecs.**

## Video Files

### Required Codec
- **Video Codec**: H.264 (libx264) - REQUIRED for browser compatibility
- **Audio Codec**: AAC or copy original (most browsers support various audio codecs)
- **Container**: MP4 (with moov atom at beginning for progressive playback)

### Why This Matters
- OpenCV's default `mp4v` (MPEG-4 Simple Profile) is **NOT supported** by web browsers
- Browsers require H.264 for reliable MP4 playback
- Even with correct codec, videos need `+faststart` flag to move metadata to beginning for streaming

### Implementation Checklist
- [ ] VideoWriter uses H.264 codec (or is re-encoded to H.264)
- [ ] Videos are optimized with `+faststart` flag (moov atom at beginning)
- [ ] Test video playback directly in browser (file:// URL) to verify compatibility
- [ ] Check codec with `ffprobe` before deployment

### Example: Optimizing Video for Web
```python
# Use ffmpeg to re-encode to H.264 and optimize for streaming
subprocess.run([
    'ffmpeg', '-i', input_video,
    '-c:v', 'libx264',      # H.264 codec (browser-compatible)
    '-preset', 'fast',       # Speed/quality balance
    '-crf', '23',            # Quality (18-28 range)
    '-c:a', 'copy',          # Copy audio (or use '-c:a aac' if needed)
    '-movflags', '+faststart',  # Move metadata to beginning
    output_video, '-y'
])
```

## Other Media Types

### Images
- **Formats**: PNG, JPEG, WebP (best browser support)
- **SVG**: Supported but verify rendering
- Avoid proprietary formats

### Audio
- **Formats**: MP3, AAC, OGG (for broader compatibility)
- **Container**: MP4 (M4A) or MP3

### Documents
- **PDF**: Universal browser support
- **Text**: Plain text, HTML, JSON

## Testing Checklist

Before deploying any web app that generates media files:

1. **Codec Verification**: Use `ffprobe` to check codec
2. **Direct Browser Test**: Open file directly in browser (file:// URL)
3. **Network Tab**: Verify HTTP 206 (Partial Content) responses for streaming
4. **Multiple Browsers**: Test in Chrome, Firefox, Safari
5. **Mobile**: Test on iOS Safari and Android Chrome if relevant

## Common Mistakes to Avoid

❌ **Don't**: Use OpenCV's default `mp4v` codec
✅ **Do**: Use H.264 or re-encode to H.264

❌ **Don't**: Assume faststart is automatic
✅ **Do**: Always use `+faststart` flag for MP4 videos

❌ **Don't**: Skip browser testing
✅ **Do**: Test in actual browsers before deployment

## Resources

- [HTML5 Video Codec Support](https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Video_codecs)
- [FFmpeg H.264 Encoding Guide](https://trac.ffmpeg.org/wiki/Encode/H.264)
- [Browser Video Format Support](https://caniuse.com/?cats=CSS,HTML5,Multimedia)

