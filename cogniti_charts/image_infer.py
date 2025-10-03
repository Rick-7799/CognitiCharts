import numpy as np, joblib
from .utils import CLASS_NAMES
from .image_data import preprocess_image
def predict_image(img_path_or_file, framework="tf", img_size=224):
    if framework=="tf":
        import tensorflow as tf
        m=tf.keras.models.load_model("models/tf_image_model.keras", compile=False)
        arr=preprocess_image(img_path_or_file,img_size); p=m.predict(arr[None,...],verbose=0)[0]
    else:
        import torch
        from .image_models_torch import TorchImageCNN
        state=joblib.load("models/torch_image_model.pt")
        net=TorchImageCNN(num_classes=len(CLASS_NAMES)); net.load_state_dict(state); net.eval()
        arr=preprocess_image(img_path_or_file,img_size).transpose(2,0,1)
        with torch.no_grad(): import torch as T; logits=net(T.from_numpy(arr[None,...]).float()); p=T.softmax(logits,dim=1).numpy()[0]
    idx=int(np.argmax(p)); return {"label_index":idx, "label": CLASS_NAMES[idx],
                                   "probs":{"Breakout":float(p[0]),"Consolidation":float(p[1]),"Reversal":float(p[2])}}
