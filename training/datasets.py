"""
Real Dataset Loaders for AI Content Detection
Replace the dummy datasets with these once you have labeled data.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer
from PIL import Image
import json
from pathlib import Path
import numpy as np
from scipy import fftpack


def add_fft_channel(image_tensor):
    """Convert RGB tensor to 4-channel tensor with FFT magnitude as 4th channel.
    
    Args:
        image_tensor: Shape (3, H, W) - normalized RGB image
        
    Returns:
        4-channel tensor: (RGB + FFT magnitude)
    """
    # Denormalize to 0-1 range for FFT processing
    rgb = image_tensor.numpy()
    
    # Convert to grayscale for FFT (rgb is shape (3, H, W))
    # Using weighted average of RGB channels
    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    
    # Compute FFT magnitude
    fft_vals = fftpack.fft2(gray)
    fft_magnitude = np.abs(fft_vals)
    
    # Log scale for better visualization
    fft_magnitude = np.log(fft_magnitude + 1)
    
    # Normalize FFT to 0-1
    fft_min = fft_magnitude.min()
    fft_max = fft_magnitude.max()
    if fft_max > fft_min:
        fft_magnitude = (fft_magnitude - fft_min) / (fft_max - fft_min)
    else:
        fft_magnitude = np.zeros_like(fft_magnitude)
    
    # Stack as 4-channel
    fft_channel = torch.from_numpy(fft_magnitude).unsqueeze(0).float()
    return torch.cat([image_tensor, fft_channel], dim=0)


class ImageDataset(Dataset):
    """
    Load images from a directory structure:
    
    Flat structure:
    data/images/
        ai_generated/
            img1.jpg
            img2.png
            ...
        real/
            photo1.jpg
            photo2.png
            ...
    
    Nested structure (with subdirectories):
    data/
        ai/
            category1/
                img1.jpg
            category2/
                img2.jpg
        Real/
            category1/
                img1.jpg
            category2/
                img2.jpg
    
    Or from a JSON manifest:
    [
        {"path": "path/to/img1.jpg", "label": 0},  # 0 = AI
        {"path": "path/to/img2.jpg", "label": 1},  # 1 = Real
        ...
    ]
    """
    
    def __init__(self, root_dir=None, manifest_path=None, img_size=224, transform=None, nested=False):
        self.img_size = img_size
        self.samples = []
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load from directory structure
        if root_dir:
            root = Path(root_dir)
            # Try multiple folder name patterns
            ai_folders = ['ai', 'ai_generated', 'AI']
            real_folders = ['Real', 'real', 'not_ai']
            
            for label, folder_list in enumerate([ai_folders, real_folders]):
                for folder_name in folder_list:
                    path = root / folder_name
                    if path.exists():
                        if nested:
                            # Nested structure: recurse into subdirectories
                            for img_path in path.rglob('*'):
                                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']:
                                    self.samples.append({'path': str(img_path), 'label': label})
                        else:
                            # Flat structure: images directly in folder
                            for img_path in path.glob('*'):
                                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']:
                                    self.samples.append({'path': str(img_path), 'label': label})
                        break  # Found valid folder, stop trying alternatives
        
        # Load from manifest JSON
        elif manifest_path:
            with open(manifest_path) as f:
                self.samples = json.load(f)
        
        print(f"Loaded {len(self.samples)} images (AI: {sum(1 for s in self.samples if s['label'] == 0)}, Real: {sum(1 for s in self.samples if s['label'] == 1)})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert('RGB')
        img = self.transform(img)  # Converts to tensor and normalizes
        # Add FFT channel (4th channel)
        img = add_fft_channel(img)
        label = torch.tensor(sample['label'], dtype=torch.long)
        return img, label


class TextDataset(Dataset):
    """
    Load text from a JSON file:
    [
        {"text": "This is AI-generated text...", "label": 0},
        {"text": "This is human-written text...", "label": 1},
        ...
    ]
    
    Or from separate files:
    data/text/
        ai_generated.txt  (one sample per line)
        real.txt          (one sample per line)
    """
    
    def __init__(self, json_path=None, ai_file=None, real_file=None, 
                 max_length=128, model_name="distilbert-base-uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.samples = []
        
        # Load from JSON
        if json_path:
            with open(json_path) as f:
                self.samples = json.load(f)
        
        # Load from separate files
        elif ai_file and real_file:
            with open(ai_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append({'text': line, 'label': 0})
            
            with open(real_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append({'text': line, 'label': 1})
        
        print(f"Loaded {len(self.samples)} text samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        label = sample['label']
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# DATA COLLECTION TIPS
# ============================================================================

"""
IMAGES:
-------
AI-generated sources:
- DALL-E: https://labs.openai.com
- Midjourney: https://www.midjourney.com
- Stable Diffusion: https://stablediffusionweb.com
- Artbreeder: https://www.artbreeder.com

Real photo sources:
- Unsplash: https://unsplash.com/developers (free API)
- Pexels: https://www.pexels.com/api/
- Your own photos
- Creative Commons licensed photos

Aim for: ~500-1000 images per class to start


TEXT:
-----
AI-generated sources:
- ChatGPT outputs (various prompts)
- Claude outputs
- Other LLM outputs

Real text sources:
- News articles (via APIs: NewsAPI, Guardian API)
- Reddit comments (via PRAW)
- Book excerpts (Project Gutenberg)
- Wikipedia articles
- Academic papers

Aim for: ~1000-2000 samples per class to start


LABELING:
---------
1. Start with clear AI vs real samples
2. As you improve, add edge cases:
   - AI-edited human text
   - Human-edited AI text
   - Hybrid content
3. Consider multi-class later (AI, human, hybrid, uncertain)
"""
