import argparse, os, numpy as np, joblib
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from .image_data import generate_synthetic_charts, preprocess_image
from .image_models_tf import make_image_model
from .image_models_torch import TorchImageCNN
from .utils import CLASS_NAMES
from .shap_utils import save_image_shap_grid
import shap

def train_tf(imgs, ys, img_size, class_weight):
    import tensorflow as tf
    os.makedirs("models", exist_ok=True)
    X=np.stack([preprocess_image(im,img_size) for im in imgs]); y=ys
    m=make_image_model(img_size, num_classes=len(CLASS_NAMES))
    es=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)
    m.fit(X,y,validation_split=0.2,epochs=20,batch_size=64,callbacks=[es],verbose=2,class_weight=class_weight)
    m.save("models/tf_image_model.keras")
        # SHAP for a small subset
    try:
        X = np.stack([preprocess_image(im, img_size) for im in imgs])
        bg = X[:4]
        sm = X[:2]
        explainer = shap.DeepExplainer(m, bg)
        sv = explainer.shap_values(sm)
        from .shap_utils import save_image_shap_grid
        save_image_shap_grid(sv, sm, path="models/shap_image_summary_tf.png")
    except Exception as e:
        print("Image SHAP skipped:", e)
    print(classification_report(y, m.predict(X,verbose=0).argmax(1), target_names=CLASS_NAMES))

def train_torch(imgs, ys, img_size, class_weight):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    os.makedirs("models", exist_ok=True)
    X=np.stack([preprocess_image(im,img_size) for im in imgs],axis=0).transpose(0,3,1,2); y=ys
    dev="cuda" if torch.cuda.is_available() else "cpu"; net=TorchImageCNN(num_classes=len(CLASS_NAMES)).to(dev)
    ds=TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    dl=DataLoader(ds,batch_size=64,shuffle=True)
    cw=torch.tensor([class_weight.get(i,1.0) for i in range(len(CLASS_NAMES))],dtype=torch.float32,device=dev)
    opt=torch.optim.Adam(net.parameters(),lr=7.5e-4); loss_fn=nn.CrossEntropyLoss(weight=cw)
    best=float("inf"); best_state=None
    for epoch in range(15):
        net.train(); run=0.0
        for xb,yb in dl:
            xb,yb=xb.to(dev), yb.to(dev); opt.zero_grad(); loss=loss_fn(net(xb), yb); loss.backward(); opt.step(); run+=loss.item()*len(yb)
        run/=len(ds); 
        if run<best: best=run; best_state={k:v.cpu() for k,v in net.state_dict().items()}
    if best_state: net.load_state_dict(best_state)
    joblib.dump(net.state_dict(),"models/torch_image_model.pt")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", default="sample_data/sample_prices.csv")
    ap.add_argument("--framework", choices=["tf","torch"], default="tf")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--per_class", type=int, default=100)
    ap.add_argument("--img_size", type=int, default=224)
    args=ap.parse_args()

    imgs, ys = generate_synthetic_charts(args.csv, lookback=args.lookback, per_class=args.per_class)
    classes=np.unique(ys); cw=compute_class_weight(class_weight="balanced", classes=classes, y=ys)
    class_weight={int(c):float(w) for c,w in zip(classes,cw)}
    mean=np.mean(list(class_weight.values()))
    for k in class_weight: class_weight[k]=float(np.clip(class_weight[k]/mean,0.6,2.0))
    print("Image class weights:", class_weight)
    if args.framework=="tf": train_tf(imgs, ys, args.img_size, class_weight)
    else: train_torch(imgs, ys, args.img_size, class_weight)

if __name__=="__main__": main()
