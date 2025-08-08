let originalText = '';
let predictedData = null;

const inputViewer = createViewer(document.getElementById('inputViewer'));
const resultViewer = createViewer(document.getElementById('resultViewer'));

function createViewer(container) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(0, 0, 2);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  return { scene, camera, renderer, controls };
}

function animate() {
  requestAnimationFrame(animate);
  inputViewer.renderer.render(inputViewer.scene, inputViewer.camera);
  inputViewer.controls.update();
  resultViewer.renderer.render(resultViewer.scene, resultViewer.camera);
  resultViewer.controls.update();
}
animate();

function parsePointCloud(text, withLabels = false) {
  const lines = text.trim().split(/\n+/);
  const positions = [];
  const semantics = [];
  const instances = [];
  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts.length < 3) continue;
    positions.push(parseFloat(parts[0]), parseFloat(parts[1]), parseFloat(parts[2]));
    if (withLabels) {
      semantics.push(parts[3]);
      instances.push(parts[4]);
    }
  }
  return { positions, semantics, instances };
}

function centerAndScale(positions) {
  const count = positions.length / 3;
  const arr = new Float32Array(positions);
  const box = new THREE.Box3();
  const vec = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    vec.set(arr[3*i], arr[3*i+1], arr[3*i+2]);
    box.expandByPoint(vec);
  }
  const center = new THREE.Vector3();
  box.getCenter(center);
  const size = new THREE.Vector3();
  box.getSize(size);
  const scale = 1 / Math.max(size.x, size.y, size.z);
  for (let i = 0; i < count; i++) {
    arr[3*i] = (arr[3*i] - center.x) * scale;
    arr[3*i+1] = (arr[3*i+1] - center.y) * scale;
    arr[3*i+2] = (arr[3*i+2] - center.z) * scale;
  }
  return arr;
}

function showInputPointCloud(text, file) {
  const data = parsePointCloud(text);
  const positions = centerAndScale(data.positions);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({ size: parseFloat(pointSize.value), color: 0x555555 });
  const points = new THREE.Points(geom, material);
  inputViewer.scene.clear();
  inputViewer.scene.add(points);
  inputViewer.points = points;
  document.getElementById('fileInfo').textContent = `${file.name} | ${(file.size/1024).toFixed(1)} KB | ${positions.length/3} points`;
  inferBtn.disabled = false;
}

function showResultPointCloud(data) {
  const positions = centerAndScale(data.positions);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const colors = new Float32Array((positions.length));
  const semanticColors = {};
  const instanceColors = {};
  const semanticCounts = {};
  const instanceCounts = {};
  const grey = new THREE.Color(0xaaaaaa);
  for (let i = 0; i < data.semantics.length; i++) {
    const s = data.semantics[i];
    const inst = data.instances[i];
    if (!semanticColors[s]) {
      const c = new THREE.Color().setHSL((Object.keys(semanticColors).length * 0.618) % 1, 0.5, 0.6);
      semanticColors[s] = c;
    }
    if (!instanceColors[inst]) {
      const c = new THREE.Color().setHSL((Object.keys(instanceColors).length * 0.618) % 1, 0.7, 0.5);
      instanceColors[inst] = c;
    }
    semanticCounts[s] = (semanticCounts[s] || 0) + 1;
    instanceCounts[inst] = (instanceCounts[inst] || 0) + 1;
    const c = semanticColors[s];
    colors[3*i] = c.r;
    colors[3*i+1] = c.g;
    colors[3*i+2] = c.b;
  }
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({ size: parseFloat(pointSize.value), vertexColors: true });
  const points = new THREE.Points(geom, material);
  resultViewer.scene.clear();
  resultViewer.scene.add(points);
  resultViewer.points = points;
  resultViewer.semanticColors = semanticColors;
  resultViewer.instanceColors = instanceColors;
  resultViewer.semantics = data.semantics;
  resultViewer.instances = data.instances;
  resultViewer.grey = grey;
  populateFilters(semanticCounts, instanceCounts);
  updateStats(semanticCounts, instanceCounts);
}

function populateFilters(semanticCounts, instanceCounts) {
  const semSel = document.getElementById('semanticFilter');
  semSel.innerHTML = '<option value="all">All</option>';
  Object.keys(semanticCounts).forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = k;
    semSel.appendChild(opt);
  });
  const instSel = document.getElementById('instanceFilter');
  instSel.innerHTML = '<option value="all">All</option>';
  Object.keys(instanceCounts).forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = k;
    instSel.appendChild(opt);
  });
}

function updateStats(semanticCounts, instanceCounts) {
  let html = '<strong>Semantic Counts</strong><ul>';
  for (const [k,v] of Object.entries(semanticCounts)) {
    html += `<li>${k}: ${v}</li>`;
  }
  html += '</ul><strong>Instance Counts</strong><ul>';
  for (const [k,v] of Object.entries(instanceCounts)) {
    html += `<li>${k}: ${v}</li>`;
  }
  html += '</ul>';
  document.getElementById('stats').innerHTML = html;
}

function applyFilters() {
  if (!resultViewer.points) return;
  const semVal = document.getElementById('semanticFilter').value;
  const instVal = document.getElementById('instanceFilter').value;
  const colors = resultViewer.points.geometry.attributes.color.array;
  for (let i = 0; i < resultViewer.semantics.length; i++) {
    const s = resultViewer.semantics[i];
    const inst = resultViewer.instances[i];
    let c;
    let visible = true;
    if (semVal !== 'all' && s !== semVal) visible = false;
    if (instVal !== 'all' && inst !== instVal) visible = false;
    if (visible) {
      if (instVal !== 'all') {
        c = resultViewer.instanceColors[inst];
      } else {
        c = resultViewer.semanticColors[s];
      }
    } else {
      c = resultViewer.grey;
    }
    colors[3*i] = c.r;
    colors[3*i+1] = c.g;
    colors[3*i+2] = c.b;
  }
  resultViewer.points.geometry.attributes.color.needsUpdate = true;
}

document.getElementById('semanticFilter').addEventListener('change', applyFilters);
document.getElementById('instanceFilter').addEventListener('change', applyFilters);

const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('inputZone');
const inferBtn = document.getElementById('inferBtn');
const pointSize = document.getElementById('pointSize');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  readFile(file);
});
fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  readFile(file);
});

function readFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    originalText = e.target.result;
    dropZone.style.display = 'none';
    showInputPointCloud(originalText, file);
  };
  reader.readAsText(file);
}

inferBtn.addEventListener('click', () => {
  fetch('/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: originalText
  }).then(r => r.text()).then(t => {
    predictedData = parsePointCloud(t, true);
    showResultPointCloud(predictedData);
  });
});

pointSize.addEventListener('input', e => {
  const size = parseFloat(e.target.value);
  if (inputViewer.points) inputViewer.points.material.size = size;
  if (resultViewer.points) resultViewer.points.material.size = size;
});

document.getElementById('resetView').addEventListener('click', () => {
  inputViewer.controls.reset();
  resultViewer.controls.reset();
});
