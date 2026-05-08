# GoldMine Watch — Product Roadmap

A functional roadmap: each stage is a working product that detects mining activity, but with increasing intelligence and accuracy.

**Philosophy:** Ship a working detector at every stage. Each stage is a complete, usable product. You can stop at any stage and still have value.

---

## Stage 1: The Rule-Based Detector

**Product:** A script that detects likely mining areas using spectral thresholds.

**What it does:**
- Downloads a Sentinel-2 image for your area and date.
- Computes spectral indices: **NDVI** (vegetation loss) and **BSI** (bare soil exposure).
- Flags pixels where NDVI is very low AND BSI is very high.
- Groups flagged pixels into polygons and exports them as a GeoPackage.

**Why it works:** Mining clears vegetation and exposes bare soil. These two indices catch that signature surprisingly well.

**Input:** A date range and a bounding box.
**Output:** A GeoPackage with polygons labeled as "potential mining disturbance."

**Success criteria:**
You open the output in QGIS. Most real mining sites from your labels are covered by the polygons, plus some false positives (bare rivers, roads). You can explain every detection: "This is red because it has no vegetation and lots of bare soil."

**Limitation:** It also flags roads, river sandbars, and burned fields. It has no concept of "shape" or "pattern."

---

## Stage 2: The Visual AI Detector

**Product:** A small AI model that looks at satellite images the way a human does — by pattern and shape.

**What it does:**
- Cuts the satellite image into small RGB patches.
- Feeds them to a lightweight image-segmentation model (tiny U-Net, MobileNet backbone).
- The model learns to recognize the visual texture and geometry of mining sites from your labeled examples.
- Outputs a probability map: "This patch looks like a mining site."

**Why it upgrades Stage 1:**
Instead of relying on hard spectral rules, it learns from examples. It can distinguish a mining clearing from a river sandbar if they look different in shape and texture.

**Input:** RGB patches + your labeled polygons.
**Output:** A heatmap and a GeoPackage with detected polygons, now ranked by model confidence.

**Success criteria:**
You compare Stage 1 and Stage 2 side-by-side in QGIS. Stage 2 has fewer obvious false positives (fewer roads flagged). It misses some small sites but gets the big ones right.

**Limitation:** It only sees RGB. It is "colorblind" to the invisible spectral information that satellites provide.

---

## Stage 3: The Full-Spectrum AI Expert

**Product:** A full multispectral AI pipeline that combines spectral intelligence with visual pattern recognition.

**What it does:**
- Builds a cloud-free composite from multiple Sentinel-2 scenes over time.
- Uses all spectral bands: Blue, Green, Red, NIR, SWIR1, SWIR2.
- Computes indices (NDVI, BSI, NDWI) and feeds them into the model alongside raw bands.
- Runs a larger, more capable model (ResNet-34 U-Net) on 9-channel input.
- Applies proper spatial train/validation splits so metrics are honest.
- Exports probability rasters, thresholded polygons, and a ready-to-use QGIS project.

**Why it upgrades Stage 2:**
It sees everything: the visual patterns (RGB) *and* the spectral signature (NIR, SWIR). Mining sites have a distinct footprint in SWIR that RGB alone cannot see. The combination is significantly more accurate.

**Input:** A time window, an area, and labeled training data.
**Output:**
- A trained model file.
- A probability heatmap (GeoTIFF).
- Detected polygons with confidence scores (GeoPackage).
- A QGIS project with all layers styled and ready.
- A training manifest with metrics (IoU, F1, precision, recall).

**Success criteria:**
You run the full pipeline with one command. The evaluation metrics beat Stage 2. The QGIS project opens and shows clean polygons over real mining sites, with very few false positives in urban or river areas.

---

## How the stages relate

| Capability | Stage 1 (Rules) | Stage 2 (Visual AI) | Stage 3 (Full AI) |
|---|---|---|---|
| Detects bare soil / vegetation loss | Yes | Learns it | Learns it |
| Understands shape and texture | No | Yes | Yes |
| Uses infrared / SWIR bands | No | No | Yes |
| Cloud-free compositing | No | No | Yes |
| Train/val split & honest metrics | No | Minimal | Yes |
| Export to QGIS | Manual | Basic | One-click project |
| Training time | Minutes | 10–30 min | 1–2 hours |
| Accuracy | Baseline | Better | Best |

**You can stop at any stage and still have a working detector.**

- Need a quick answer for one area? Use **Stage 1**.
- Want a reusable detector that improves with more labels? Use **Stage 2**.
- Need production-grade accuracy and full evaluation? Use **Stage 3**.

---

## Suggested path

1. **Build Stage 1 first.** It takes a few hours and gives you immediate, verifiable results. You will learn a lot about your data and your labels.
2. **Build Stage 2 next.** You now have a baseline to beat. If the AI is not better than the rules, you know something is wrong.
3. **Build Stage 3 last.** By now you understand the data, the labels, and the model behavior. The full pipeline is a natural extension, not a leap of faith.
