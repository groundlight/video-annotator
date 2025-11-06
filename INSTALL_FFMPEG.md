# Installing ffmpeg for Video Optimization

## Option 1: Install Homebrew (Recommended)

Homebrew is the easiest package manager for macOS:

1. **Install Homebrew**:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   Follow the prompts. This typically takes 5-10 minutes.

2. **Install ffmpeg**:
   ```bash
   brew install ffmpeg
   ```

3. **Verify installation**:
   ```bash
   ffmpeg -version
   ```

## Option 2: Download ffmpeg Directly

If you don't want to install Homebrew, you can download a pre-built ffmpeg:

1. **Download from**:
   - https://evermeet.cx/ffmpeg/ (macOS builds)
   - Or use: https://github.com/BtbN/FFmpeg-Builds/releases

2. **Extract and add to PATH**:
   ```bash
   # Create a bin directory in your home folder
   mkdir -p ~/bin
   
   # Move ffmpeg there (adjust path to where you extracted it)
   mv /path/to/ffmpeg ~/bin/ffmpeg
   
   # Add to PATH (add this to ~/.zshrc)
   echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

## Option 3: Use Without Optimization (Temporary)

Videos will still work without optimization, but:
- Browser may need to download entire file before playback starts
- Seeking may be slower
- Preview may not work immediately

**To test videos without optimization:**
1. Videos will be generated normally (just not optimized)
2. You can manually optimize later using any ffmpeg installation
3. Or use online tools like CloudConvert to optimize MP4s

## Verify Installation

After installing, test with:
```bash
which ffmpeg
ffmpeg -version
```

The webapp will automatically detect and use ffmpeg if available.

