"""Inference and sliding-window prediction.

Modules:
    predict: Run model prediction on a single patch.
    predict_big: Sliding-window inference on large raster tiles.
    evaluate: Compute IoU, F1, precision, recall against labels.
    tiler: Split large images into overlapping tiles.
    blender: Average overlapping tile predictions.
    postprocess: Threshold and convert predictions to vector polygons.
"""
