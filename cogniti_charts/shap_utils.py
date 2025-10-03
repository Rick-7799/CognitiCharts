import os
import numpy as np
import shap
import matplotlib.pyplot as plt

def _ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def save_summary_plot(shap_vals, X_2d, path="models/shap_summary.png", feature_names=None):
    """Save a SHAP summary plot for 2D inputs (N, D)."""
    _ensure_dir(path)
    plt.figure()
    shap.summary_plot(shap_vals, X_2d, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

def _kernel_explainer_over_windows(predict_fn, X_bg, X_sm, out_path, feature_names=None, nsamples=100):
    """
    Generic, reliable SHAP via KernelExplainer on flattened windows.
    X_bg, X_sm are 3D: (N, L, F). We flatten to (N, L*F), wrap predict_fn to reshape back.
    """
    L = X_bg.shape[1]
    F = X_bg.shape[2]
    bg2 = X_bg.reshape(X_bg.shape[0], -1).astype("float32")
    sm2 = X_sm.reshape(X_sm.shape[0], -1).astype("float32")

    def f(z2d):
        z3d = z2d.reshape((-1, L, F)).astype("float32")
        return predict_fn(z3d)  

  
    explainer = shap.Explainer(f, bg2[:50], algorithm="permutation")  
    sv_list = explainer(sm2[:50], max_evals=nsamples)  

    
    sv = sv_list.values 
    sv_mag = np.sum(np.abs(sv), axis=2)  
    return save_summary_plot(sv_mag, sm2[:50], path=out_path, feature_names=feature_names)

def explain_tf_timeseries(model, X_bg, X_sample, feature_names=None, out_path="models/shap_summary_tf.png"):
    """
    Robust SHAP for TF time-series using a kernel/permutation explainer.
    Works even in eager mode and arbitrary Keras models.
    """
    def predict_fn(x3d):
        
        y = model.predict(x3d, verbose=0)
        return y
    return _kernel_explainer_over_windows(predict_fn, X_bg, X_sample, out_path, feature_names=feature_names)

def explain_torch_timeseries(model, X_bg, X_sample, feature_names=None, out_path="models/shap_summary_torch.png"):
    """
    Robust SHAP for Torch time-series using a kernel/permutation explainer.
    """
    import torch
    model.eval()

    def predict_fn(x3d):
        with torch.no_grad():
            t = torch.from_numpy(x3d).float()
            logits = model(t)
            p = torch.softmax(logits, dim=1).numpy()
        return p

    return _kernel_explainer_over_windows(predict_fn, X_bg, X_sample, out_path, feature_names=feature_names)

def save_image_shap_grid(shap_values, imgs, path="models/shap_image_summary.png"):
    """Save a grid for image SHAP using shap.image_plot (TF image path)."""
    _ensure_dir(path)
    plt.figure()
    shap.image_plot(shap_values, imgs, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
