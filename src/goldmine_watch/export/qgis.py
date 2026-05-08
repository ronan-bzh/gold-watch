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
    """Append a minimal raster layer definition to the project XML."""
    layer = SubElement(parent, "maplayer", {"type": "raster"})
    SubElement(layer, "id").text = layer_id
    SubElement(layer, "datasource").text = str(source_path)
    SubElement(layer, "layername").text = layer_name
    SubElement(layer, "srs")
    SubElement(layer, "provider").text = "gdal"
    SubElement(layer, "type").text = "1"  # raster

    # Minimal renderer so QGIS knows it's multi-band RGB or single-band
    rasterrenderer = SubElement(
        layer,
        "rasterrenderer",
        {"type": "multibandcolor", "opacity": "1", "alphaBand": "-1"},
    )
    SubElement(rasterrenderer, "rasterTransparency")


def _add_vector_layer(
    parent: Element,
    layer_id: str,
    layer_name: str,
    source_path: Path,
) -> None:
    """Append a minimal vector layer definition to the project XML."""
    layer = SubElement(parent, "maplayer", {"type": "vector"})
    SubElement(layer, "id").text = layer_id
    SubElement(layer, "datasource").text = str(source_path)
    SubElement(layer, "layername").text = layer_name
    SubElement(layer, "srs")
    SubElement(layer, "provider").text = "ogr"
    SubElement(layer, "geometry").text = "Polygon"

    # Simple outline renderer
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
    SubElement(layer_prop, "prop", {"k": "outline_color", "v": "255,0,0,255"})  # red outline
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
