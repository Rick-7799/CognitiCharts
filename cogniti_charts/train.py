import argparse, os, numpy as np, pandas as pd, joblib
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from .data import load_prices, label_patterns, train_test_split_time
from .features import build_features
from .utils import window_stack, CLASS_NAMES
from .shap_utils import explain_tf_timeseries, explain_torch_timeseries
from . import models_tf, models_torch

def build_xy(df: pd.DataFrame, lookback: int):
    feats = build_features(df)
    X = window_stack(feats.values, lookback)
    y = df["label"].values[lookback - 1 :]
    return X.astype("float32"), y.astype("int64")

def train_tf(Xtr, ytr, Xval, yval, class_weight):
    import tensorflow as tf
    os.makedirs("models", exist_ok=True)
    model = models_tf.make_tf_model(
        seq_len=Xtr.shape[1], num_features=Xtr.shape[2], num_classes=len(CLASS_NAMES)
    )
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)
    ckpt = tf.keras.callbacks.ModelCheckpoint("models/best_tf.keras", monitor="val_loss", save_best_only=True)
    model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=25, batch_size=64,
        callbacks=[es, rlr, ckpt],
        verbose=2, class_weight=class_weight,
    )
    model.save("models/tf_model.keras")
    return model

def train_torch(Xtr, ytr, Xval, yval):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = models_torch.TorchCNN1D(num_features=Xtr.shape[2], num_classes=len(CLASS_NAMES)).to(device)
    classes, counts = np.unique(ytr, return_counts=True)
    raw = (counts.max() / counts).astype(np.float32)
    raw = raw / raw.mean()
    raw = np.clip(raw, 0.7, 1.8).astype(np.float32)
    class_weight = torch.tensor(raw, dtype=torch.float32, device=device)
    print("Torch class weights (normalized+clipped):", dict(zip(classes.tolist(), raw.tolist())))
    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))
    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=128, shuffle=False)
    opt = torch.optim.Adam(net.parameters(), lr=7.5e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    best_acc, best_state = 0.0, None
    for epoch in range(20):
        net.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        net.eval(); correct=0; total=0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = net(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        acc = correct / max(1, total)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in net.state_dict().items()}
    if best_state:
        net.load_state_dict(best_state)
    os.makedirs("models", exist_ok=True)
    joblib.dump(net.state_dict(), "models/torch_model.pt")
    return net.cpu()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sample_data/sample_prices.csv")
    ap.add_argument("--framework", choices=["tf", "torch"], default="tf")
    ap.add_argument("--lookback", type=int, default=60)
    args = ap.parse_args()

    
    df = load_prices(args.csv)
    df = label_patterns(df, lookback=args.lookback)
    tr, va, te = train_test_split_time(df, test_ratio=0.2, val_ratio=0.1)

  
    Xtr, ytr = build_xy(tr, args.lookback)
    Xva, yva = build_xy(pd.concat([tr.iloc[-args.lookback+1:], va]), args.lookback)
    Xte, yte = build_xy(pd.concat([va.iloc[-args.lookback+1:], te]), args.lookback)

  
    classes = np.unique(ytr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    mean_w = np.mean(list(class_weight.values()))
    for k in class_weight:
        class_weight[k] = float(np.clip(class_weight[k]/mean_w, 0.6, 2.0))
    print("Class weights (normalized+clipped):", class_weight)

    
    if args.framework == "tf":
        model = train_tf(Xtr, ytr, Xva, yva, class_weight=class_weight)
        probs = model.predict(Xte, verbose=0)
        yhat = probs.argmax(1)
    else:
        net = train_torch(Xtr, ytr, Xva, yva)
        import torch
        from .models_torch import TorchCNN1D
        net_eval = TorchCNN1D(num_features=Xte.shape[2], num_classes=len(CLASS_NAMES))
        net_eval.load_state_dict(net.state_dict()); net_eval.eval()
        with torch.no_grad():
            logits = net_eval(torch.from_numpy(Xte))
            yhat = logits.argmax(1).numpy()

   
    try:
        bg_size = min(64, Xtr.shape[0])
        sm_size = min(128, Xva.shape[0])
        X_bg = Xtr[:bg_size]
        X_sm = Xva[:sm_size]
        if args.framework == "tf":
            explain_tf_timeseries(model, X_bg, X_sm, feature_names=None, out_path="models/shap_summary_tf.png")
        else:
            explain_torch_timeseries(net_eval, X_bg, X_sm, feature_names=None, out_path="models/shap_summary_torch.png")
    except Exception as _e:
        print("SHAP generation skipped:", _e)

   
    print(classification_report(yte, yhat, labels=[0,1,2], target_names=CLASS_NAMES, zero_division=0))

if __name__ == "__main__":
    main()
