import sys,base64,io, json, time, pathlib
from PIL import Image
import numpy as np
sys.path.insert(0,'native')
import echo_host_V2 as host
# create a synthetic photo-like image (gradient + details)
arr = np.zeros((300,400,3),dtype=np.uint8)
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        arr[i,j]=[(i+j)%256, (2*i+j)%256, (i+2*j)%256]
img=Image.fromarray(arr,'RGB')
buf=io.BytesIO(); img.save(buf, format='JPEG', quality=90); b=buf.getvalue()
dataurl='data:image/jpeg;base64,'+base64.b64encode(b).decode('ascii')
items=[{'id':'img-test','modality':'image','source':'img','url':dataurl,'width':img.width,'height':img.height}]
envelope={'requestId':'test-1','payload':{'items':items,'model':{}}}
res=host.classify_items(envelope)
print(json.dumps(res,indent=2))
