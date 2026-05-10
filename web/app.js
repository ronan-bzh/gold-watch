// GoldMine Watch — Web Map
// Initialize map centered on French Guiana
const map = L.map('map').setView([4.0, -53.0], 7);

// Base layer: OpenStreetMap
const osmLayer = L.tileLayer(
  'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  { attribution: '&copy; OpenStreetMap contributors' }
).addTo(map);

// Satellite imagery overlay (using ESRI World Imagery as a fallback)
const satelliteLayer = L.tileLayer(
  'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {
    attribution: 'Esri, Maxar, Earthstar Geographics',
    opacity: 0.7,
  }
);

// Sentinel-2 overlay via dynamic XYZ tile server
// Dezoom is automatic — Leaflet requests lower z tiles, server resamples on the fly
const copernicusLayer = L.tileLayer('/tiles/{z}/{x}/{y}.png', {
  attribution: 'Sentinel-2 / Copernicus',
  maxZoom: 14,
  opacity: 1.0,
});

// Layer groups
let detectionsLayer = L.layerGroup().addTo(map);
let labelsLayer = L.layerGroup().addTo(map);

// UI elements
const thresholdSlider = document.getElementById('threshold');
const thresholdValue = document.getElementById('threshold-value');
const showLabelsCheckbox = document.getElementById('show-labels');
const showDetectionsCheckbox = document.getElementById('show-detections');
const showSatelliteCheckbox = document.getElementById('show-satellite');
const showCopernicusCheckbox = document.getElementById('show-copernicus');
const statusDiv = document.getElementById('status');

let detectionsData = [];
let labelsData = [];

function setStatus(msg, type = 'info') {
  statusDiv.textContent = msg;
  statusDiv.className = 'status ' + type;
}

function clearStatus() {
  statusDiv.textContent = '';
  statusDiv.className = 'status';
}

// Haversine formula to compute approximate area in m² for a square polygon
function polygonAreaM2(coords) {
  // coords is a GeoJSON Polygon ring: [[lon, lat], ...]
  const R = 6371000; // Earth radius in meters
  const toRad = (deg) => (deg * Math.PI) / 180;
  let area = 0;
  for (let i = 0; i < coords.length - 1; i++) {
    const lon1 = toRad(coords[i][0]);
    const lat1 = toRad(coords[i][1]);
    const lon2 = toRad(coords[i + 1][0]);
    const lat2 = toRad(coords[i + 1][1]);
    area +=
      (lon2 - lon1) *
      (2 + Math.sin(lat1) + Math.sin(lat2));
  }
  area = Math.abs((area * R * R) / 2.0);
  return area;
}

function renderDetections() {
  detectionsLayer.clearLayers();
  const threshold = parseInt(thresholdSlider.value, 10) / 100;

  let visibleCount = 0;
  detectionsData.forEach((feature) => {
    const confidence = feature.properties.confidence ?? feature.properties.pred ?? 0;
    if (confidence < threshold) return;

    const geom = feature.geometry;
    if (!geom || geom.type !== 'Polygon') return;

    const latlngs = geom.coordinates[0].map((c) => [c[1], c[0]]);
    const area = polygonAreaM2(geom.coordinates[0]);

    const poly = L.polygon(latlngs, {
      color: '#d32f2f',
      fillColor: '#d32f2f',
      fillOpacity: 0.4,
      weight: 2,
    });

    const popupContent = `
      <strong>Detection</strong><br/>
      Confidence: <strong>${(confidence * 100).toFixed(1)}%</strong><br/>
      Area: ~${area.toFixed(0)} m²
    `;
    poly.bindPopup(popupContent);
    poly.addTo(detectionsLayer);
    visibleCount++;
  });

  setStatus(`Showing ${visibleCount} detections (threshold ≥ ${(threshold * 100).toFixed(0)}%)`);
}

function renderLabels() {
  labelsLayer.clearLayers();
  labelsData.forEach((feature) => {
    const geom = feature.geometry;
    if (!geom || geom.type !== 'Polygon') return;

    const latlngs = geom.coordinates[0].map((c) => [c[1], c[0]]);
    const area = polygonAreaM2(geom.coordinates[0]);
    const confidence = feature.properties.confidence ?? feature.properties.pred ?? 0;

    const poly = L.polygon(latlngs, {
      color: '#388e3c',
      fillColor: '#388e3c',
      fillOpacity: 0.2,
      weight: 1.5,
      dashArray: '4,4',
    });

    const popupContent = `
      <strong>Original Label</strong><br/>
      Confidence: <strong>${(confidence * 100).toFixed(1)}%</strong><br/>
      Area: ~${area.toFixed(0)} m²
    `;
    poly.bindPopup(popupContent);
    poly.addTo(labelsLayer);
  });
}

function updateVisibility() {
  if (showLabelsCheckbox.checked) {
    if (!map.hasLayer(labelsLayer)) map.addLayer(labelsLayer);
  } else {
    if (map.hasLayer(labelsLayer)) map.removeLayer(labelsLayer);
  }

  if (showDetectionsCheckbox.checked) {
    if (!map.hasLayer(detectionsLayer)) map.addLayer(detectionsLayer);
  } else {
    if (map.hasLayer(detectionsLayer)) map.removeLayer(detectionsLayer);
  }

  if (showSatelliteCheckbox.checked) {
    if (!map.hasLayer(satelliteLayer)) map.addLayer(satelliteLayer);
  } else {
    if (map.hasLayer(satelliteLayer)) map.removeLayer(satelliteLayer);
  }

  if (showCopernicusCheckbox.checked) {
    if (!map.hasLayer(copernicusLayer)) map.addLayer(copernicusLayer);
  } else {
    if (map.hasLayer(copernicusLayer)) map.removeLayer(copernicusLayer);
  }
}

async function loadGeoJSON(path, onSuccess, label) {
  try {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    const features = data.features || [];
    onSuccess(features);
    console.log(`Loaded ${features.length} features from ${path}`);
  } catch (err) {
    console.warn(`Failed to load ${label}:`, err.message);
    setStatus(`Note: ${label} not available (${err.message})`, 'warning');
  }
}

// Event listeners
thresholdSlider.addEventListener('input', () => {
  thresholdValue.textContent = thresholdSlider.value;
  renderDetections();
});

showLabelsCheckbox.addEventListener('change', updateVisibility);
showDetectionsCheckbox.addEventListener('change', updateVisibility);
showSatelliteCheckbox.addEventListener('change', updateVisibility);
showCopernicusCheckbox.addEventListener('change', updateVisibility);

// Load data
(async function init() {
  clearStatus();
  setStatus('Loading map data...');

  await loadGeoJSON('data/detections.geojson', (features) => {
    detectionsData = features;
    renderDetections();
  }, 'Detections');

  await loadGeoJSON('data/labels.geojson', (features) => {
    labelsData = features;
    renderLabels();
  }, 'Labels');

  updateVisibility();
  clearStatus();
})();
