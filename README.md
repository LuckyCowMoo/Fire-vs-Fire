# Fire v Fire

A browser extension that detects AI-generated images and text in real-time using deep learning classifiers.

## Overview

Fire v Fire is a Firefox/Chrome extension that automatically scans web pages for AI-generated content and applies visual indicators (blur, borders, badges) to flag suspicious content. It uses native Python classifiers with PyTorch for on-device inference.

## Features

- **Real-time Detection**: Automatically scans images and text as pages load
- **Multi-Model Ensemble**: Supports multiple classifiers with weighted averaging
- **Visual Indicators**: Configurable blur, borders, badges, and strikethrough effects
- **Smart Caching**: Remembers classifications to avoid redundant processing
- **Lazy Loading**: Unloads models when idle to conserve VRAM
- **Context Menu**: Right-click images/text to manually toggle blur
- **Streaming Results**: Progressive UI updates as classifications complete

## Architecture

### Browser Extension (JavaScript)
- **background.js**: Manages native messaging and classification requests
- **content.js**: Scans DOM, extracts content, applies visual effects
- **options.js**: Settings UI for thresholds, styles, and ensemble configuration

### Native Host (Python)
- **echo_host_V2.py**: Native messaging bridge between browser and classifiers
- **model_registry.py**: Manages classifier loading/unloading
- **classifiers/**: PyTorch model implementations (ResNet-50, ConvNeXt, etc.)

## Installation

### Prerequisites
- Python 3.11+ with GPU support (torch-directml for AMD/Intel, CUDA for NVIDIA)
- Firefox or Chrome browser
- Windows (registry-based native host registration)

### Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Register native host**:
   - Firefox: Run `install-firefox.ps1` or import `register-firefox-host.reg`
   - Chrome: Import `register-chrome-host.reg`

3. **Load extension**:
   - Firefox: `about:debugging` → Load Temporary Add-on → Select `manifest.json`
   - Chrome: `chrome://extensions` → Load unpacked → Select project folder

## Configuration

### Ensemble Configuration
Edit `native/ensemble_config.json` to customize classifier ensembles:

```json
{
  "classifiers": ["resnet50_fft"],
  "weights": [1.0],
  "miniBatchSize": 10,
  "lazyLoad": true
}
```

### Extension Settings
Access via browser toolbar icon:
- **Detection Thresholds**: Adjust AI confidence thresholds (0-100%)
- **Visual Effects**: Configure blur amount, border color, badge display
- **Performance**: Mini-batch size, lazy loading, classification delay
- **Filters**: Minimum image size, text length, alt-text requirements

## Models

Supported classifiers (in `native/classifiers/`):
- **resnet50_fft**: ResNet-50 with FFT artifact detection
- **convnext_large**: ConvNeXt-Large for high-accuracy classification
- **text**: Text-based AI detection (placeholder)

Models are loaded on-demand and cached in VRAM. Lazy loading automatically unloads models after 15 minutes of inactivity.

## Usage

1. **Automatic Scanning**: Browse normally - extension scans pages automatically
2. **Manual Blur**: Right-click image/text → "Toggle AI blur"
3. **View Results**: Hover over flagged content to reveal original
4. **Adjust Settings**: Click extension icon to open options panel

## Development

### Project Structure
```
Fire v Fire/
├── background/          # Background script (native messaging)
├── content/            # Content script (DOM scanning, visual effects)
├── options/            # Settings UI
├── native/             # Python native host
│   ├── classifiers/    # PyTorch model implementations
│   ├── echo_host_V2.py # Main native messaging loop
│   └── model_registry.py # Classifier registry
├── icons/              # Extension icons
└── manifest.json       # Extension manifest
```

### Building
```bash
npm run build  # Creates .zip for distribution
```

### Debugging
- Browser console: Extension logs prefixed with `[AI Detector]`
- Native host logs: `native_host_v2.log`
- Enable verbose logging in extension settings

## Performance

- **Image Classification**: ~100-500ms per image (GPU-dependent)
- **Text Classification**: ~50-200ms per section
- **Memory Usage**: ~2-4GB VRAM per loaded model
- **Cache Hit Rate**: ~80-90% on typical browsing

## Limitations

- Requires local GPU for acceptable performance
- Only supports Firefox/Chrome on Windows
- Models may produce false positives on artistic/stylized content
- Text detection is experimental and less accurate than image detection

## License

Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

Copyright (c) 2026 Leah (Luke) Armstrong (LuckyCowMoo)

See [LICENSE](LICENSE) for full terms.

## Credits

- PyTorch and torchvision for deep learning framework
- ResNet-50 architecture from Microsoft Research
- ConvNeXt architecture from Meta AI Research
- Browser extension template from Mozilla

## Support

For issues or questions, check the logs:
1. Browser console (F12) for extension errors
2. `native_host_v2.log` for classifier errors
3. Enable verbose logging in settings for detailed diagnostics
