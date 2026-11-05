"""One-off patch for fvf_hydra_classifier.py (UTF-8 safe)."""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "native" / "classifiers" / "fvf_hydra_classifier.py"
text = path.read_text(encoding="utf-8")

old_ckpt = '''        for rel in (
            "checkpoints/hydra/last.pt",
            "checkpoints/hydra/epoch_001.pt",
        ):'''

new_ckpt = '''        for rel in (
            "checkpoints/hydra_webscrape3/last.pt",
            "checkpoints/hydra_webscrape3/epoch_001.pt",
            "checkpoints/hydra/last.pt",
            "checkpoints/hydra/epoch_001.pt",
        ):'''

if old_ckpt in text:
    text = text.replace(old_ckpt, new_ckpt)

init_old = "        self._ai_class_index = 1"
init_new = """        self._ai_class_index = 1
        self._inference_crops = int(os.environ.get("FVF_HYDRA_CROPS", "5"))
        self._score_agg = os.environ.get("FVF_HYDRA_AGG", "max").strip().lower()"""
if init_old in text and "_inference_crops" not in text:
    text = text.replace(init_old, init_new)

crops_old = """                crops = self._take_crops(
                    img,
                    self._crop_policy,
                    requested=None,
                    training_random_one=False,
                )"""
crops_new = """                crops = self._take_crops(
                    img,
                    self._crop_policy,
                    requested=self._inference_crops,
                    training_random_one=False,
                )"""
if crops_old in text:
    text = text.replace(crops_old, crops_new)

pipe_old = """                    out = self._pipeline.extract(rgb)
                    ch = out.tensor if hasattr(out, "tensor") else out"""
pipe_new = "                    ch = self._pipeline(rgb)"
if pipe_old in text:
    text = text.replace(pipe_old, pipe_new)

agg_old = """                p = probs_cpu[row_idxs]
                scores.append(float(p.mean().item()))"""
agg_new = """                p = probs_cpu[row_idxs]
                if self._score_agg == "mean":
                    scores.append(float(p.mean().item()))
                else:
                    scores.append(float(p.max().item()))"""
if agg_old in text:
    text = text.replace(agg_old, agg_new)

path.write_text(text, encoding="utf-8")
print("patched", path)
