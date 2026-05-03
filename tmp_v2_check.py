import sys
sys.path.insert(0, r'C:\Users\lukea\Desktop\Stuff\School\Year 3\Project\Fire v Fire\native')
from classifiers.convnext_large_artifact_classifier_V2 import ConvNeXtLargeArtifactV2Classifier

c = ConvNeXtLargeArtifactV2Classifier()
path = c._find_model_path()
sys.stdout.write(f'Model path: {path}\n')
sys.stdout.write(f'Exists: {path.exists() if path else False}\n')
sys.stdout.write(f'ai_class_index default: {c._ai_class_index}\n')
sys.stdout.flush()
