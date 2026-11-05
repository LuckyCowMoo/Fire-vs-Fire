"""
Base classifier interface for modular AI detection models.

All classifiers must implement this interface to be compatible with
the model registry and orchestrator system.

Supports multiple modalities:
- 'image': PIL Image objects
- 'text': Text strings
- 'audio': Audio data (future)
- 'video': Video data (future)

QUICK CHECKLIST FOR NEW CLASSIFIERS:
===================================
1. Import BaseClassifier
2. Implement ALL 6 abstract methods (get_supported_modalities, load_model, 
    preprocess_batch, classify_batch, get_device_info, get_model_name)
3. For ML models: Use lazy imports in _ensure_imports() to avoid startup delays
4. For GPU models: Set self.device in load_model() using _try_cuda_then_directml()
5. In preprocess_batch: Return (batch_tensor, valid_indices) - indices must match batch_tensor order
6. In classify_batch: Return List[float] matching batch length (scores or -1.0 for errors)
7. Set self._is_loaded = True after successful load_model()
8. Use _log_to_stderr() for all debug messages (avoids corrupting native messaging)
9. Optionally override get_label() to provide a human-readable badge label

COMMON GOTCHAS:
- Don't use print() directly - use sys.stderr or _log_to_stderr()
- valid_indices in preprocess_batch() MUST match batch_tensor order
- Scores must be in [0.0, 1.0] where 1.0 = detected content, 0.0 = not detected content, -1.0 = error
- Always check is_loaded() before inference; process_batch() handles this
- If model_path is None, provide a sensible default in __init__
- If get_label() is not overridden, the UI falls back to "AI generated"
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import sys


class BaseClassifier(ABC):
    """Abstract base class for AI detection classifiers (multi-modal)."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Optional path to model weights. If None, uses default.
        """
        self.model_path = model_path
        self.model = None
        self.device = None
        self._is_loaded = False
    
    @abstractmethod
    def get_supported_modalities(self) -> Set[str]:
        """
        Get modalities this classifier supports.
        
        Returns:
            Set of modalities (e.g., {'image'}, {'text'}, {'image', 'text'})
        """
        pass
    
    @abstractmethod
    def load_model(self) -> tuple[bool, Optional[str]]:
        """
        Load the model weights and prepare for inference.
        
        Returns:
            Tuple of (success: bool, error_reason: Optional[str])
            - (True, None) if successful
            - (False, "error_reason") if failed
        """
        pass
    
    @abstractmethod
    def preprocess_batch(self, inputs: List[Any], modality: str) -> tuple[Any, List[int]]:
        """
        Preprocess a batch of inputs for inference.
        
        Args:
            inputs: List of inputs (PIL Images, text strings, audio data, etc.)
            modality: Type of input ('image', 'text', 'audio', 'video')
        
        Returns:
            Tuple of (batch_tensor, valid_indices)
            - batch_tensor: Preprocessed tensor ready for model (e.g., torch.Tensor)
            - valid_indices: Indices of successfully preprocessed inputs
        """
        pass
    
    @abstractmethod
    def classify_batch(self, batch_tensor: Any) -> List[float]:
        """
        Run inference on a preprocessed batch.
        
        Args:
            batch_tensor: Preprocessed batch from preprocess_batch()
        
        Returns:
            List of scores in [0.0, 1.0] range where:
            - 1.0 = definitely AI-generated
            - 0.0 = definitely real/human-made
            - -1.0 = classification error
        """
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the device being used for inference.
        
        Returns:
            Dict with keys:
            - 'device': str (e.g., 'cuda', 'cpu', 'privateuseone:0')
            - 'name': str (e.g., 'AMD Radeon RX 7900 XTX')
            - 'backend': str (e.g., 'DirectML', 'CUDA', 'CPU')
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get a human-readable name for this classifier.
        
        Returns:
            Model name (e.g., 'ResNet50-FFT', 'EfficientNet-B3')
        """
        pass

    def get_label(self) -> str:
        """
        Get the human-readable badge label for this classifier.

        Returns:
            Label shown in the UI badge (e.g., 'AI generated', 'Spider').
            Override this in subclasses to provide a more specific label.

        Notes:
            This is optional. Existing classifiers inherit the default label
            without needing any code changes.
        """
        return "AI generated"
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._is_loaded
    
    def unload_model(self):
        """
        Unload model from memory (for lazy loading).
        Override this method if you need custom cleanup.
        """
        self.model = None
        self._is_loaded = False
    
    def process_batch(self, inputs: List[Any], modality: str) -> List[float]:
        """
        Convenience method that combines preprocessing and classification.
        
        Args:
            inputs: List of inputs (PIL Images, text strings, audio data, etc.)
            modality: Type of input ('image', 'text', 'audio', 'video')
        
        Returns:
            List of scores (same length as input items)
            Returns -1.0 for items that failed preprocessing or if model loading fails
        """
        if not self.is_loaded():
            success, error = self.load_model()
            if not success:
                return [-1.0] * len(inputs)
        
        batch_tensor, valid_indices = self.preprocess_batch(inputs, modality)
        
        if not valid_indices:
            return [-1.0] * len(inputs)
        
        scores = self.classify_batch(batch_tensor)
        
        # Fill in results, marking failed inputs as -1.0
        results = [-1.0] * len(inputs)
        for idx, score in zip(valid_indices, scores):
            results[idx] = score
        
        return results

    # ============================================================================
    # OPTIONAL HELPER METHODS (use these to avoid code duplication)
    # ============================================================================
    
    def _log_to_stderr(self, message: str) -> None:
        """
        Safe logging to stderr (doesn't corrupt native messaging output).
        Use this instead of print() for all debug/status messages.
        
        Args:
            message: Message to log
        """
        try:
            print(message, file=sys.stderr, flush=True)
        except Exception:
            pass
    
    def _try_cuda_then_directml(self, lib_torch=None) -> Any:
        """
        Convenience method for setting up GPU device with proper fallback.
        
        Tries DirectML first (AMD/Intel GPU on Windows), then CUDA, then CPU.
        Sets self.device and self._device_info if they exist.
        
        Args:
            lib_torch: torch module (if None, tries to import dynamically)
        
        Returns:
            Device object (torch.device or equivalent)
        
        Example:
            self.device = self._try_cuda_then_directml(self.torch)
        """
        if lib_torch is None:
            try:
                import torch
                lib_torch = torch
            except ImportError:
                self._log_to_stderr("torch not available")
                return None
        
        # Try DirectML (Windows GPU)
        try:
            import torch_directml
            device = torch_directml.device()
            device_name = torch_directml.device_name(0)
            if hasattr(self, '_device_info'):
                self._device_info = {
                    'device': 'privateuseone:0',
                    'name': device_name,
                    'backend': 'DirectML'
                }
            self._log_to_stderr(f"✓ Using GPU (DirectML): {device_name}")
            return device
        except ImportError:
            pass
        except Exception as e:
            self._log_to_stderr(f"DirectML failed: {e}")
        
        # Try CUDA
        if lib_torch.cuda.is_available():
            device = lib_torch.device('cuda')
            try:
                device_name = lib_torch.cuda.get_device_name(0)
            except Exception:
                device_name = 'CUDA GPU'
            if hasattr(self, '_device_info'):
                self._device_info = {
                    'device': 'cuda',
                    'name': device_name,
                    'backend': 'CUDA'
                }
            self._log_to_stderr(f"✓ Using GPU (CUDA): {device_name}")
            return device
        
        # Fallback to CPU
        device = lib_torch.device('cpu')
        if hasattr(self, '_device_info'):
            self._device_info = {
                'device': 'cpu',
                'name': 'CPU',
                'backend': 'CPU'
            }
        self._log_to_stderr("⚠ Using CPU (GPU not available)")
        return device
    
    def _ensure_imports_safely(self, import_dict: Dict[str, str]) -> bool:
        """
        Safely import multiple libraries with clear error reporting.
        
        Args:
            import_dict: Dict mapping attribute names to module names
                        e.g., {'torch': 'torch', 'np': 'numpy', 'Image': 'PIL.Image'}
        
        Returns:
            True if all imports succeeded, False otherwise
        
        Example:
            success = self._ensure_imports_safely({
                'torch': 'torch',
                'nn': 'torch.nn',
                'np': 'numpy',
            })
            if success:
                self.torch, self.nn, self.np = [module_instances]
        """
        try:
            for attr_name, module_path in import_dict.items():
                try:
                    parts = module_path.split('.')
                    module = __import__(module_path)
                    for part in parts[1:]:
                        module = getattr(module, part)
                    setattr(self, attr_name, module)
                except ImportError:
                    self._log_to_stderr(f"Failed to import {module_path}")
                    return False
            return True
        except Exception as e:
            self._log_to_stderr(f"Import error: {e}")
            return False
