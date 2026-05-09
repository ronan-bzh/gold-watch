"""Create QGIS project files (.qgz) without the QGIS Python API.

A .qgz file is a ZIP archive containing a single .qgs XML file.
We build the XML manually so the project remains dependency-free.
"""

import zipfile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring


def _add_raster_layer(
    parent: Element,
    layer_id: str,
    layer_name: str,
    source_path: Path,
) -> None:
    """Append a minimal multi-band raster layer definition to the project XML."""
    layer = SubElement(parent, "maplayer", {"type": "raster"})
    SubElement(layer, "id").text = layer_id
    SubElement(layer, "datasource").text = str(source_path)
    SubElement(layer, "layername").text = layer_name
    SubElement(layer, "srs")
    SubElement(layer, "provider").text = "gdal"
    SubElement(layer, "type").text = "1"  # raster

    # Minimal renderer so QGIS knows it's multi-band RGB
    rasterrenderer = SubElement(
        layer,
        "rasterrenderer",
        {"type": "multibandcolor", "opacity": "1", "alphaBand": "-1"},
    )
    SubElement(rasterrenderer, "rasterTransparency")


def _add_probability_raster_layer(
    parent: Element,
    layer_id: str,
    layer_name: str,
    source_path: Path,
) -> None:
    """Append a single-band pseudocolor raster layer for probability heatmaps."""
    layer = SubElement(parent, "maplayer", {"type": "raster"})
    SubElement(layer, "id").text = layer_id
    SubElement(layer, "datasource").text = str(source_path)
    SubElement(layer, "layername").text = layer_name
    SubElement(layer, "srs")
    SubElement(layer, "provider").text = "gdal"
    SubElement(layer, "type").text = "1"  # raster

    rasterrenderer = SubElement(
        layer,
        "rasterrenderer",
        {
            "type": "singlebandpseudocolor",
            "opacity": "1",
            "alphaBand": "-1",
            "band": "1",
            "classificationMax": "1",
            "classificationMin": "0",
        },
    )
    SubElement(rasterrenderer, "rasterTransparency")
    min_max = SubElement(rasterrenderer, "minMaxOrigin")
    SubElement(min_max, "limits").text = "MinMax"
    SubElement(min_max, "extent").text = "WholeRaster"
    SubElement(min_max, "statAccuracy").text = "Estimated"

    rastershader = SubElement(rasterrenderer, "rastershader")
    colorrampshader = SubElement(
        rastershader,
        "colorrampshader",
        {"classificationMode": "2", "clip": "0", "colorRampType": "INTERPOLATED"},
    )
    colorramp = SubElement(
        colorrampshader,
        "colorramp",
        {"name": "[source]", "type": "gradient"},
    )
    SubElement(colorramp, "prop", {"k": "color1", "v": "255,255,255,255"})
    SubElement(colorramp, "prop", {"k": "color2", "v": "255,0,0,255"})
    SubElement(colorramp, "prop", {"k": "discrete", "v": "0"})

    # Gradient stops: white (0) -> yellow (0.5) -> red (1)
    SubElement(
        colorrampshader,
        "item",
        {
            "alpha": "255",
            "color": "#ffffff",
            "label": "0.0000",
            "value": "0",
        },
    )
    SubElement(
        colorrampshader,
        "item",
        {
            "alpha": "255",
            "color": "#ffff00",
            "label": "0.5000",
            "value": "0.5",
        },
    )
    SubElement(
        colorrampshader,
        "item",
        {
            "alpha": "255",
            "color": "#ff0000",
            "label": "1.0000",
            "value": "1",
        },
    )


def _add_vector_layer(
    parent: Element,
    layer_id: str,
    layer_name: str,
    source_path: Path,
    outline_color: str = "255,0,0,255",
) -> None:
    """Append a minimal vector layer definition to the project XML.

    Args:
        parent: XML element to append the layer to.
        layer_id: Unique identifier for the layer.
        layer_name: Display name for the layer.
        source_path: Path to the vector data source.
        outline_color: RGBA color string for the outline (default red).
    """
    layer = SubElement(parent, "maplayer", {"type": "vector"})
    SubElement(layer, "id").text = layer_id
    SubElement(layer, "datasource").text = str(source_path)
    SubElement(layer, "layername").text = layer_name
    SubElement(layer, "srs")
    SubElement(layer, "provider").text = "ogr"
    SubElement(layer, "geometry").text = "Polygon"

    # Simple outline renderer with configurable color
    renderer = SubElement(
        layer,
        "renderer-v2",
        {"type": "singleSymbol", "symbollevels": "0", "enableorderby": "0"},
    )
    symbols = SubElement(renderer, "symbols")
    symbol = SubElement(
        symbols,
        "symbol",
        {"name": "0", "type": "fill", "alpha": "1", "clip_to_extent": "1"},
    )
    layer_prop = SubElement(symbol, "layer", {"pass": "0", "class": "SimpleFill", "locked": "0"})
    SubElement(layer_prop, "prop", {"k": "color", "v": "0,0,0,0"})  # transparent fill
    SubElement(layer_prop, "prop", {"k": "outline_color", "v": outline_color})
    SubElement(layer_prop, "prop", {"k": "outline_width", "v": "0.5"})


def create_qgis_project(
    image_path: Path,
    prediction_path: Path,
    polygons_path: Path,
    output_project_path: Path,
) -> Path:
    """Create a .qgz QGIS project file referencing the three input layers.

    Layers (in order):
      - Satellite image (multi-band raster)
      - Probability heatmap (single-band raster)
      - Detected polygons (vector, outlined in red)

    Args:
        image_path: Path to the satellite image GeoTIFF.
        prediction_path: Path to the probability GeoTIFF.
        polygons_path: Path to the polygons GeoPackage.
        output_project_path: Path to save the .qgz project.

    Returns:
        Path to the saved .qgz file.
    """
    output_project_path = Path(output_project_path)
    output_project_path.parent.mkdir(parents=True, exist_ok=True)

    # Build minimal QGIS project XML
    qgis = Element("qgis", {"projectname": "", "version": "3.28.0-Firenze"})
    projectlayers = SubElement(qgis, "projectlayers")

    # Use relative paths for portability
    base = output_project_path.parent

    def _rel(p: Path) -> str:
        try:
            return str(Path(p).relative_to(base))
        except ValueError:
            return str(Path(p).resolve())

    _add_raster_layer(
        projectlayers,
        "satellite_image",
        "Satellite Image",
        Path(_rel(image_path)),
    )
    _add_raster_layer(
        projectlayers,
        "probability_heatmap",
        "Probability Heatmap",
        Path(_rel(prediction_path)),
    )
    _add_vector_layer(
        projectlayers,
        "detected_polygons",
        "Detected Polygons",
        Path(_rel(polygons_path)),
    )

    qgs_name = output_project_path.stem + ".qgs"
    xml_bytes = tostring(qgis, encoding="utf-8", xml_declaration=True)

    with zipfile.ZipFile(output_project_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(qgs_name, xml_bytes)

    return output_project_path


def create_qgis_project_full(
    mosaic_path: Path,
    detections_path: Path,
    labels_path: Path,
    output_project_path: Path,
    image_path: Path | None = None,
) -> Path:
    """Create a .qgz QGIS project file for the full territory.

    Layers (in order):
      - Satellite composite image (multi-band raster, optional)
      - Mosaic probability heatmap (single-band raster, pseudocolor)
      - Detections (vector, outlined in red)
      - Labels (vector, outlined in green)

    Args:
        mosaic_path: Path to the mosaic GeoTIFF (probability heatmap).
        detections_path: Path to the detections GeoJSON or GeoPackage.
        labels_path: Path to the labels GeoJSON or GeoPackage.
        output_project_path: Path to save the .qgz project.
        image_path: Optional path to an RGB composite GeoTIFF.

    Returns:
        Path to the saved .qgz file.
    """
    output_project_path = Path(output_project_path)
    output_project_path.parent.mkdir(parents=True, exist_ok=True)

    # Build minimal QGIS project XML
    qgis = Element("qgis", {"projectname": "", "version": "3.28.0-Firenze"})
    projectlayers = SubElement(qgis, "projectlayers")

    # Use relative paths for portability
    base = output_project_path.parent

    def _rel(p: Path) -> str:
        try:
            return str(Path(p).relative_to(base))
        except ValueError:
            return str(Path(p).resolve())

    if image_path is not None:
        _add_raster_layer(
            projectlayers,
            "satellite_composite",
            "Satellite Composite",
            Path(_rel(image_path)),
        )

    _add_probability_raster_layer(
        projectlayers,
        "probability_heatmap",
        "Probability Heatmap",
        Path(_rel(mosaic_path)),
    )
    _add_vector_layer(
        projectlayers,
        "detections",
        "Detections",
        Path(_rel(detections_path)),
        outline_color="255,0,0,255",  # red
    )
    _add_vector_layer(
        projectlayers,
        "labels",
        "Labels",
        Path(_rel(labels_path)),
        outline_color="0,255,0,255",  # green
    )

    qgs_name = output_project_path.stem + ".qgs"
    xml_bytes = tostring(qgis, encoding="utf-8", xml_declaration=True)

    with zipfile.ZipFile(output_project_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(qgs_name, xml_bytes)

    return output_project_path
