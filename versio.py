import pathlib
import cv2
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

IMGDIR = pathlib.Path.home() / "Downloads" / "Sabatier" / "images"
PATDIR = pathlib.Path(".") / "samples"
# 0: versio antiqua
# 1: vulgata nova
# 2: notae ad
# 3: caput

data = {}
labels = []
templates = []

# parse annotation files
for annot in sorted(PATDIR.glob("image-????.txt")):
    image = cv2.imread(str(IMGDIR / f"{annot.stem}.png"), cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    cols = ["cls", "xc", "yc", "bw", "bh"]
    df = pd.read_csv(annot, sep=" ", header=None, names=cols)

    # from YOLO to cartesian coordinates
    df["xmin"] = np.floor((df["xc"] - df["bw"] / 2) * w).astype(int)
    df["xmax"] = np.ceil((df["xc"] + df["bw"] / 2) * w).astype(int)
    df["ymin"] = np.floor((df["yc"] - df["bh"] / 2) * h).astype(int)
    df["ymax"] = np.ceil((df["yc"] + df["bh"] / 2) * h).astype(int)

    for rec in df.itertuples():
        # crop roughly the region of interest
        roi: np.ndarray = image[rec.ymin:rec.ymax, rec.xmin:rec.xmax]

        # threshold to isolate title
        ret, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # find tight bounding box around title
        x_min = min(cv2.boundingRect(c)[0] for c in contours)
        y_min = min(cv2.boundingRect(c)[1] for c in contours)
        x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
        y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)

        # normalize and resize cropped title image
        cropped = roi[y_min:y_max, x_min:x_max].astype(np.float32) / 255.
        cropped =  (cropped - np.mean(cropped)) / (np.std(cropped))
        cropped = cv2.resize(cropped, (224, 24), interpolation=cv2.INTER_AREA)
        labels.append(rec.cls)
        templates.append(cropped)

labels = np.array(labels)
templates = np.array(templates)

def cosine_similarity(roi, templates, labels):
    dot = np.sum(templates * roi, axis=(1, 2))  # (N,)
    template_norms = np.linalg.norm(templates, axis=(1, 2))  # (N,)
    roi_norm = np.linalg.norm(roi)

    scores = dot / (template_norms * roi_norm + 1e-8)
    return labels[np.argmax(scores)], np.max(scores)

results = []
for pngfile in tqdm(sorted(IMGDIR.glob("image-00??.png"))):
    img = cv2.imread(pngfile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annoted = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = gray[y:y+h, x:x+w]        
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cropped = thresh.astype(np.float32) / 255.
        cropped =  (roi - np.mean(roi)) / (np.std(roi) + 1e-6)
        cropped = cv2.resize(cropped, (224, 24), interpolation=cv2.INTER_AREA)
        idx, score = cosine_similarity(cropped, templates, labels)

        if score > 0.5:
            cv2.rectangle(annoted, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annoted, f"{idx}: {score:.03f}", (x-80, y+(h//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            results.append((pngfile.name, idx, score, x, y, w, h))
    cv2.imwrite(pngfile.name, annoted)

df1 = pd.DataFrame(results, columns=["image", "cls", "score", "x", "y", "w", "h"])
df1.to_csv("results.csv", index=False)
