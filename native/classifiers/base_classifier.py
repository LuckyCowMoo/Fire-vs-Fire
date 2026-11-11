"""
Base classifier interface for modular content classification and AI detection.

All classifiers placed in native/classifiers/ must inherit from BaseClassifier
to be automatically discovered, registered, and managed by the Native Host.

Supports multi-modal pipelines:
- 'image': PIL Image objects
- 'text': Text strings / token sequences
- 'audio': Audio waveforms / spectrograms
- 'video': Video frames / streams

CORE CONTRACT:
=============
Each subclass is an independent Python class. Hardware acceleration, framework
choice (PyTorch, ONNX Runtime, TensorRT, OpenVINO, TensorFlow, pure Python heuristics),
device placement, and runtime management are entirely up to the classifier implementation.

To integrate with the orchestrator, implement the 6 abstract methods:
1. get_supported_modalities() -> Set[str]
   Return supported modalities (e.g. {'image'}, {'text'}).
2. load_model() -> tuple[bool, Optional[str]]
   Initialize your model / engine. Return (True, None) on success or (False, "reason") on failure.
   Set self._is_loaded = True when ready.
3. preprocess_batch(inputs, modality) -> tuple[Any, List[int]]
   Transform raw inputs into model input format.
   Return (batch_tensor, valid_indices) where valid_indices maps batch items back to inputs.
4. classify_batch(batch_tensor) -> List[float]
   Execute inference. Return a list of float scores in [0.0, 1.0] matching batch size,
   or -1.0 for processing errors.
5. get_device_info() -> Dict[str, Any]
   Diagnostic metadata for the extension UI (e.g. {'device': 'cuda', 'name': 'NVIDIA RTX 4080'}).
6. get_model_name() -> str
   Human-readable model name (e.g. 'FvF Heimdall 04').

OPTIONAL EXTENSIONS:
====================
- get_label() -> str:
  Badge text shown in the browser UI (e.g. "AI generated", "Spider", "Deepfake").
  Defaults to "AI generated".
- REQUIRED_PACKAGES: List[str]:
  PyPI packages needed by this classifier (e.g. ["onnxruntime-directml>=1.18.0"]).
  The Native Host checks these on load and can auto-install missing packages.
- unload_model():
  Called when freeing resources/VRAM during inactivity or modality switches.

CRITICAL PROTOCOL RULE:
=======================
Never use print() or write to sys.stdout!
The browser extension communicates with the Native Host via binary length-prefixed
JSON on stdout. Any raw stdout output corrupts the protocol framing.
Always use self._log_to_stderr() or print(..., file=sys.stderr) for debug logging.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import sys


class BaseClassifier(ABC):
    """Abstract base class for AI detection classifiers (multi-modal)."""
    
    # Class-level list of required PyPI packages for this classifier.
    # Third-party / custom classifiers can declare any extra libraries they need:
    # e.g. REQUIRED_PACKAGES = ["timm>=0.9.0", "transformers", "scikit-image"]
    # The native host checks dependencies in-memory on load and auto-installs missing packages.
    REQUIRED_PACKAGES: List[str] = []
    
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
        if hasattr(self, "ort_session"):
            self.ort_session = None
        self._is_loaded = False
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
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

    def process_item(self, item_input: Any, modality: str) -> float:
        """
        Convenience method for single-item inference (preprocess + classify 1 item).
        Subclasses may override this if they have a specialized single-item path,
        otherwise it delegates cleanly to process_batch([item_input], modality).

        Args:
            item_input: Single input item (PIL Image, text string, audio/video data)
            modality: Type of input ('image', 'text', 'audio', 'video')

        Returns:
            Float score in [0.0, 1.0] or -1.0 on error
        """
        if item_input is None or (modality == 'text' and item_input == ''):
            return -1.0
        scores = self.process_batch([item_input], modality)
        if not scores:
            return -1.0
        return float(scores[0])

    def get_estimated_memory_mb(self) -> float:
        """
        Return the estimated RAM/VRAM footprint of this classifier in MB when loaded.
        Used by the Native Host ModelMemoryManager to decide how many models can stay
        warm concurrently in memory ("keep the engine running") without exhausting
        hardware resources on lower-end consumer systems.

        Subclasses can override this or set `ESTIMATED_MEMORY_MB` on the class.
        Defaults to 2.2x the model weight file size on disk (plus 150 MB runtime buffer),
        or 350.0 MB if the weight file size is unknown.
        """
        explicit = getattr(self, "ESTIMATED_MEMORY_MB", None)
        if isinstance(explicit, (int, float)) and explicit > 0:
            return float(explicit)
        try:
            mpath = getattr(self, "model_path", None)
            if mpath is None and hasattr(self, "_find_checkpoint") and callable(self._find_checkpoint):
                mpath = self._find_checkpoint()
            if mpath and Path(mpath).is_file():
                size_mb = Path(mpath).stat().st_size / (1024 * 1024)
                return round(max(150.0, size_mb * 2.2 + 150.0), 1)
        except Exception:
            pass
        return 350.0

    @staticmethod
    def get_system_memory_info() -> Dict[str, float]:
        """
        Probe current system RAM and GPU VRAM availability in MB.
        Works zero-dependency across Windows, Linux, and macOS.

        Returns:
            Dict with keys:
            - 'ram_total_mb': float
            - 'ram_avail_mb': float
            - 'vram_total_mb': float (0.0 if no dedicated CUDA GPU queried)
            - 'vram_avail_mb': float (0.0 if no dedicated CUDA GPU queried)
        """
        ram_total_mb = 8192.0
        ram_avail_mb = 4096.0
        vram_total_mb = 0.0
        vram_avail_mb = 0.0

        try:
            if sys.platform == "win32":
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    ram_total_mb = stat.ullTotalPhys / (1024 * 1024)
                    ram_avail_mb = stat.ullAvailPhys / (1024 * 1024)
            elif hasattr(sys, "platform") and Path("/proc/meminfo").is_file():
                meminfo = {}
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val_kb = int(parts[1].strip().split()[0])
                            meminfo[key] = val_kb / 1024.0
                ram_total_mb = meminfo.get("MemTotal", ram_total_mb)
                ram_avail_mb = meminfo.get("MemAvailable", meminfo.get("MemFree", ram_avail_mb))
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                vram_avail_mb = free_b / (1024 * 1024)
                vram_total_mb = total_b / (1024 * 1024)
        except Exception:
            pass

        return {
            "ram_total_mb": round(ram_total_mb, 1),
            "ram_avail_mb": round(ram_avail_mb, 1),
            "vram_total_mb": round(vram_total_mb, 1),
            "vram_avail_mb": round(vram_avail_mb, 1),
        }

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
        Optional helper method for PyTorch-based models on Windows/macOS/Linux.
        
        Attempts device discovery in order: CUDA -> Apple Silicon MPS -> DirectML -> CPU.
        Provided purely as an optional utility; classifiers are free to manage their own devices.
        
        Args:
            lib_torch: torch module (if None, attempts dynamic import)
        
        Returns:
            Device object (torch.device or equivalent)
        """
        if lib_torch is None:
            try:
                import torch
                lib_torch = torch
            except ImportError:
                self._log_to_stderr("torch not available")
                return None
        
        # Try CUDA first (NVIDIA GPUs)
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

        # Try Apple Silicon MPS (Metal Performance Shaders / Apple Neural Engine via MPSGraph)
        if hasattr(lib_torch.backends, 'mps') and lib_torch.backends.mps.is_available():
            try:
                device = lib_torch.device('mps')
                if hasattr(self, '_device_info'):
                    self._device_info = {
                        'device': 'mps',
                        'name': 'Apple Silicon (Metal/ANE)',
                        'backend': 'MPS'
                    }
                self._log_to_stderr("✓ Using Apple Silicon GPU/ANE (MPS)")
                return device
            except Exception as e:
                self._log_to_stderr(f"Apple Silicon MPS failed: {e}")

        # Try DirectML (AMD/Intel GPU & Windows NPU)
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
    

    @staticmethod
    def get_preferred_onnx_providers() -> List[str]:
        """Global hardware execution provider hierarchy for ONNX Runtime.
        Ordered strictly by capability, performance, and efficiency:
          1. CUDAExecutionProvider      - NVIDIA GPUs (RTX 5080, 4090, 3060, etc.)
          2. CoreMLExecutionProvider    - Apple Silicon (Apple Neural Engine + Metal GPU on macOS)
          3. QNNExecutionProvider       - Qualcomm Snapdragon X Elite Hexagon NPU (45 TOPS)
          4. OpenVINOExecutionProvider  - Intel Core Ultra NPU & Arc GPUs
          5. DmlExecutionProvider       - Windows DirectML (Qualcomm Adreno GPU, AMD Radeon, Intel)
          6. CPUExecutionProvider       - Multithreaded CPU fallback
        """
        return [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]

    @staticmethod
    def get_onnx_provider_options(chosen_providers: List[str]) -> List[Dict[str, Any]]:
        """Configure provider-specific accelerator options."""
        options = []
        for p in chosen_providers:
            if p == "QNNExecutionProvider":
                options.append({"backend_path": "QnnHtp.dll", "profiling_level": "off"})
            elif p == "CoreMLExecutionProvider":
                options.append({"coreml_flags": "0x001"})  # ANE + GPU + CPU
            else:
                options.append({})
        return options

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
