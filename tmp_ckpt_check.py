import torch, sys

path = r'C:\Users\lukea\Desktop\Stuff\School\Year 3\Project\Fire v Fire\native\classifiers\immage_classifier_V3-2_ConvNeXtLarge_Artifact_epoch0005.pt'
ckpt = torch.load(path, map_location='cpu', weights_only=False)
info = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
for k, v in info.items():
    sys.stdout.write(f'{k}: {v}\n')
sys.stdout.flush()
