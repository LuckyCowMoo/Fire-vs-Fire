"""
Model registry for dynamic classifier discovery and management.

Automatically discovers classifiers in the classifiers/ folder and provides
lazy loading, caching, and ensemble management.
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import importlib.util
import inspect

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from classifiers.base_classifier import BaseClassifier


class ModelRegistry:
    """Registry for managing AI detection classifiers."""

    def __init__(self):
        """Initialize empty registry."""
        self._classifiers: Dict[str, Type[BaseClassifier]] = {}
        self._classifier_sources: Dict[str, Path] = {}
        self._loaded_instances: Dict[str, BaseClassifier] = {}
        self._discovered = False

    def _ensure_dependencies(self, classifier_class: Type[BaseClassifier]) -> bool:
        """
        Checks if REQUIRED_PACKAGES defined on classifier_class are installed.
        Uses importlib.util.find_spec for sub-millisecond in-memory checking.
        Only invokes pip install if a dependency is actually missing.
        """
        required = getattr(classifier_class, 'REQUIRED_PACKAGES', None)
        if not required:
            return True

        missing = []
        overrides = {
            'scikit-learn': 'sklearn',
            'scikit-image': 'skimage',
            'opencv-python': 'cv2',
            'pillow': 'PIL',
            'pyyaml': 'yaml',
        }

        for pkg in required:
            base_name = pkg.split('>=')[0].split('<=')[0].split('==')[0].split('>')[0].split('<')[0].strip()
            module_name = overrides.get(base_name.lower(), base_name)
            try:
                if importlib.util.find_spec(module_name) is None:
                    missing.append(pkg)
            except Exception:
                missing.append(pkg)

        if not missing:
            return True

        print(f"[Model Registry] Auto-installing missing dependencies for {classifier_class.__name__}: {missing}", file=sys.stderr)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--default-timeout=1000", *missing])
            print(f"[Model Registry] Successfully installed {missing}", file=sys.stderr)
            return True
        except Exception as e:
            print(f"[Model Registry] Failed to install dependencies {missing}: {e}", file=sys.stderr)
            return False

    def discover_classifiers(self, force: bool = False):
        """
        Automatically discover classifier classes in classifiers/ folder.

        Searches for Python files in classifiers/ directory and imports
        any classes that inherit from BaseClassifier.
        """
        if self._discovered and not force:
            return
        if force:
            self._classifiers.clear()
            self._classifier_sources.clear()

        classifiers_dir = Path(__file__).parent / 'classifiers'
        if not classifiers_dir.exists():
            print(f"[Registry] Classifiers directory not found: {classifiers_dir}", file=sys.stderr)
            return

        # Import all .py files except __init__
        for py_file in classifiers_dir.glob('*.py'):
            if py_file.name.startswith('_'):
                continue

            # Skip the shared interface module; it's not a classifier plugin.
            # Importing it via our dynamic loader would overwrite
            # `classifiers.base_classifier` in sys.modules and can break
            # issubclass() checks due to class identity changes.
            if py_file.stem == 'base_classifier':
                continue

            module_name = py_file.stem
            try:
                # Import module
                spec = importlib.util.spec_from_file_location(
                    f"classifiers.{module_name}",
                    py_file
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not create import spec for {py_file}")
                module = importlib.util.module_from_spec(spec)
                # Register module before execution so decorators like @dataclass
                # and typing machinery can resolve module globals during import.
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)

                # Find BaseClassifier subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseClassifier) and
                        obj is not BaseClassifier and
                        obj.__module__ == f"classifiers.{module_name}"):

                        classifier_id = self._generate_id(name)
                        self._classifiers[classifier_id] = obj
                        self._classifier_sources[classifier_id] = py_file
                        print(f"[Registry] Discovered: {classifier_id} -> {name}", file=sys.stderr)

            except Exception as e:
                print(f"[Registry] Failed to import {module_name}: {e}", file=sys.stderr)

        self._discovered = True

    def _generate_id(self, class_name: str) -> str:
        """
        Generate a simple ID from class name.

        Examples:
            ResNet50FFTClassifier -> resnet50_fft
            EfficientNetB3Classifier -> efficientnet_b3
        """
        # Remove 'Classifier' suffix
        name = class_name.replace('Classifier', '')

        # Convert CamelCase to snake_case
        import re
        # Insert underscore before uppercase letter that follows lowercase/digit
        name = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', name)
        # Insert underscore between consecutive uppercase letters followed by lowercase
        # This handles "FFT" -> "FFT" but "ResNet" -> "Res_Net"
        name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)

        return name.lower()

    def list_available(self) -> List[str]:
        """
        Get list of available classifier IDs.

        Returns:
            List of classifier IDs (e.g., ['resnet50_fft', 'efficientnet_b3'])
        """
        if not self._discovered:
            self.discover_classifiers()
        return list(self._classifiers.keys())

    def list_available_details(self, force_rescan: bool = False) -> List[Dict[str, Any]]:
        """
        Get rich information about available classifiers discovered directly on disk.

        Returns:
            List of dicts with keys: id, name, label, modalities, weights_ready,
            weight_path, size_mb, download_size, doc, version, py_file
        """
        if not self._discovered or force_rescan:
            self.discover_classifiers(force=force_rescan)

        repo_root = Path(__file__).resolve().parent.parent
        classifiers_dir = Path(__file__).parent / 'classifiers'
        details = []
        for classifier_id, classifier_class in self._classifiers.items():
            weights_ready = False
            weight_path = None
            size_mb = 0.0
            download_size = "Local"
            modalities = []
            name = classifier_class.__name__
            label = 'AI generated'
            doc = (classifier_class.__doc__ or '').strip()
            version = getattr(classifier_class, 'VERSION', '1.0.0-local')
            py_path = self._classifier_sources.get(classifier_id)
            rel_py = ""
            if py_path:
                try:
                    rel_py = str(py_path.resolve().relative_to(repo_root)).replace('\\', '/')
                except Exception:
                    rel_py = str(py_path).replace('\\', '/')

            try:
                instance = classifier_class(model_path=None)
                modalities = sorted(list(instance.get_supported_modalities()))
                name = instance.get_model_name()
                label = instance.get_label()

                # Check weights
                mpath = getattr(instance, 'model_path', None)
                find_ckpt = getattr(instance, '_find_checkpoint', None)
                if callable(find_ckpt):
                    ckpt = find_ckpt()
                    if ckpt and Path(ckpt).is_file():
                        mpath = Path(ckpt)

                # Fallback: look in classifiers_dir for matching weights
                if (mpath is None or not Path(mpath).is_file()):
                    for ext_pattern in (f"{classifier_id}*.pt", f"{classifier_id}*.onnx", f"*{classifier_id}*.pt"):
                        found = list(classifiers_dir.glob(ext_pattern))
                        if found:
                            mpath = found[0]
                            break

                if mpath is not None and Path(mpath).is_file() and Path(mpath).stat().st_size > 0:
                    weights_ready = True
                    weight_path = str(Path(mpath).resolve())
                    size_bytes = Path(mpath).stat().st_size
                    size_mb = round(size_bytes / (1024 * 1024), 1)
                    download_size = f"{round(size_mb)} MB"
            except Exception as exc:
                print(f"[Registry] Error getting details for {classifier_id}: {exc}", file=sys.stderr)

            details.append({
                'id': classifier_id,
                'name': name,
                'label': label or 'AI generated',
                'modalities': modalities or ['image'],
                'weights_ready': weights_ready,
                'weight_path': weight_path,
                'size_mb': size_mb,
                'download_size': download_size,
                'doc': doc,
                'version': version,
                'py_file': rel_py,
            })

        return details

    def get_classifier(self, classifier_id: str, model_path: Optional[Path] = None,
                       lazy_load: bool = True) -> Optional[BaseClassifier]:
        """
        Get a classifier instance by ID.

        Args:
            classifier_id: Classifier identifier (e.g., 'resnet50_fft')
            model_path: Optional path to model weights
            lazy_load: If True, reuse cached instance; if False, create new instance

        Returns:
            Classifier instance or None if not found
        """
        if not self._discovered:
            self.discover_classifiers()

        if classifier_id not in self._classifiers:
            aliases = {
                "fvf_ares": "fvf_heimdall02",
                "fvf_heimdall_02": "fvf_heimdall02",
                "fvf_heimdall02": "fvf_heimdall_02",
                "fvf_heimdall_03": "fvf_heimdall03",
                "fvf_heimdall03": "fvf_heimdall_03",
                "fvf_heimdall_04": "fvf_heimdall04",
                "fvf_heimdall04": "fvf_heimdall_04",
                "fvf_heimdall_04_b": "fvf_heimdall04_b",
                "fvf_heimdall04_b": "fvf_heimdall_04_b",
                "fvf_heimdall04b": "fvf_heimdall_04_b",
                "fvf_mimir_02": "fvf_mimir",
                "fvf_mimir02": "fvf_mimir",
                "mimir": "fvf_mimir",
                "text": "fvf_mimir",
                "fvf_heimdall_04b": "fvf_heimdall_04_b",
            }
            alias_target = aliases.get(classifier_id)
            if alias_target and alias_target in self._classifiers:
                classifier_id = alias_target
            else:
                print(f"[Registry] Unknown classifier: {classifier_id}", file=sys.stderr)
                return None

        # Return cached instance if available and lazy loading enabled
        cache_key = f"{classifier_id}:{model_path}"
        if lazy_load and cache_key in self._loaded_instances:
            return self._loaded_instances[cache_key]

        # Ensure declared dependencies are present before instantiation
        classifier_class = self._classifiers[classifier_id]
        self._ensure_dependencies(classifier_class)

        # Create new instance
        try:
            instance = classifier_class(model_path=model_path)

            if lazy_load:
                self._loaded_instances[cache_key] = instance

            return instance

        except Exception as e:
            print(f"[Registry] Failed to instantiate {classifier_id}: {e}", file=sys.stderr)
            return None

    def unload_classifier(self, classifier_id: str, model_path: Optional[Path] = None):
        """
        Unload a cached classifier to free memory.

        Args:
            classifier_id: Classifier to unload
            model_path: Specific model path to unload (or None for all)
        """
        if model_path:
            cache_key = f"{classifier_id}:{model_path}"
            if cache_key in self._loaded_instances:
                self._loaded_instances[cache_key].unload_model()
                del self._loaded_instances[cache_key]
        else:
            # Unload all instances of this classifier
            to_remove = [k for k in self._loaded_instances.keys()
                        if k.startswith(f"{classifier_id}:")]
            for key in to_remove:
                self._loaded_instances[key].unload_model()
                del self._loaded_instances[key]

    def unload_all(self):
        """Unload all cached classifiers."""
        for instance in self._loaded_instances.values():
            instance.unload_model()
        self._loaded_instances.clear()

    def get_info(self, classifier_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a classifier without loading it.

        Args:
            classifier_id: Classifier ID

        Returns:
            Dict with 'name', 'class', 'loaded' status
        """
        if not self._discovered:
            self.discover_classifiers()

        if classifier_id not in self._classifiers:
            return None

        classifier_class = self._classifiers[classifier_id]
        is_loaded = any(k.startswith(f"{classifier_id}:")
                       for k in self._loaded_instances.keys())

        return {
            'id': classifier_id,
            'class': classifier_class.__name__,
            'loaded': is_loaded
        }


# Global registry instance
_global_registry = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
        _global_registry.discover_classifiers()
    return _global_registry
