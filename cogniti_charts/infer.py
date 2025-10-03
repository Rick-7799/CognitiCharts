import numpy as np, pandas as pd, joblib
from .data import load_prices
from .features import build_features
from .utils import window_stack, CLASS_NAMES
def _load_tf():
    import tensorflow as tf
    return tf.keras.models.load_model("models/tf_model.keras", compile=False)
def _load_torch(num_features):
    import torch
    from .models_torch import TorchCNN1D
    net=TorchCNN1D(num_features=num_features); state=joblib.load("models/torch_model.pt"); net.load_state_dict(state); net.eval(); return net
def predict(csv_path, lookback=60, framework="tf", min_breakout_prob=0.60, min_reversal_prob=0.55):
    df=load_prices(csv_path); X=window_stack(build_features(df).values, lookback).astype("float32")
    if framework=="tf":
        m=_load_tf(); probs=m.predict(X,verbose=0); yhat=probs.argmax(1)
    else:
        import torch; net=_load_torch(num_features=X.shape[2])
        with torch.no_grad(): logits=net(torch.from_numpy(X)); import torch as T; probs=T.softmax(logits,dim=1).numpy(); yhat=probs.argmax(1)
    for i in range(len(yhat)):
        if yhat[i]==0 and probs[i,0]<float(min_breakout_prob): yhat[i]=1
        if yhat[i]==2 and probs[i,2]<float(min_reversal_prob): yhat[i]=1
    return pd.DataFrame({"date": df["date"].iloc[lookback-1:].reset_index(drop=True),
                         "pred_class": yhat,
                         "pred_label": [CLASS_NAMES[i] for i in yhat],
                         "p_breakout": probs[:,0], "p_consolidation": probs[:,1], "p_reversal": probs[:,2]})
