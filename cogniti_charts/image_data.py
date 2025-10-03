import io, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from .data import load_prices, label_patterns
def generate_synthetic_charts(csv_path, lookback=60, per_class=100, fig_size=(4,3)):
    import numpy as np
    df=label_patterns(load_prices(csv_path), lookback=lookback)
    prices=df["close"].values; labels=df["label"].values
    xs=[]; ys=[]; count={0:0,1:0,2:0}
    for i in range(lookback-1,len(df)):
        y=int(labels[i])
        if count[y]>=per_class: continue
        window=prices[i-lookback+1:i+1]
        fig,ax=plt.subplots(figsize=fig_size); ax.plot(window,linewidth=2); ax.axis("off"); fig.tight_layout(pad=0)
        buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=100,bbox_inches="tight",pad_inches=0); plt.close(fig); buf.seek(0)
        img=Image.open(buf).convert("RGB")
        xs.append(img); ys.append(y); count[y]+=1
        if all(count[c]>=per_class for c in count): break
    return xs, np.array(ys,dtype=np.int64)
def preprocess_image(img, img_size=224):
    if not isinstance(img, Image.Image): img=Image.open(img).convert("RGB")
    img=img.resize((img_size,img_size)); import numpy as np; return np.asarray(img).astype("float32")/255.0
