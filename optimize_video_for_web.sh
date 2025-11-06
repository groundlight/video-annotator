#!/bin/bash
# Script to optimize MP4 video for web streaming
# Moves moov atom to beginning for faster playback start

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_video.mp4> [output_video.mp4]"
    echo "Optimizes MP4 for web streaming by moving moov atom to beginning"
    echo ""
    echo "This requires ffmpeg. Install with: brew install ffmpeg"
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.mp4}_optimized.mp4}"

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file not found: $INPUT"
    exit 1
fi

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is required but not installed."
    echo ""
    echo "Option 1: Install Homebrew (recommended)"
    echo "  1. Install Homebrew:"
    echo "     /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "  2. Install ffmpeg:"
    echo "     brew install ffmpeg"
    echo ""
    echo "Option 2: Download ffmpeg directly"
    echo "  See INSTALL_FFMPEG.md for instructions"
    echo ""
    echo "Option 3: Use online tools"
    echo "  https://www.online-convert.com/convert-to-mp4"
    echo "  (Select 'Optimize for web streaming' option)"
    exit 1
fi

echo "Optimizing video for web streaming..."
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo ""

# Use ffmpeg to optimize for web streaming
# -c:v libx264: Re-encode to H.264 (browser-compatible codec)
# -preset fast: Good balance between speed and file size
# -crf 23: High quality (18-28 is reasonable range)
# -c:a copy: Copy audio without re-encoding (faster)
# -movflags +faststart: Move moov atom to beginning for progressive playback
ffmpeg -i "$INPUT" -c:v libx264 -preset fast -crf 23 -c:a copy -movflags +faststart "$OUTPUT" -y 2>&1 | grep -E "(Duration|Stream|Output|error|H\.264|libx264)" || true

if [ $? -eq 0 ] && [ -f "$OUTPUT" ]; then
    echo ""
    echo "✓ Video optimized successfully!"
    echo "Optimized file: $OUTPUT"
    echo ""
    echo "Original size: $(du -h "$INPUT" | cut -f1)"
    echo "Optimized size: $(du -h "$OUTPUT" | cut -f1)"
    echo ""
    echo "You can now use the optimized video for web testing."
else
    echo ""
    echo "Error: Failed to optimize video"
    exit 1
fi

