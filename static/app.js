// 以 ES Module 方式引入 three 与 OrbitControls（固定版本）
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';

/* ========== 工具函数 ========== */

// —— 每点独立透明度的 ShaderMaterial（用于右侧高亮降透明度，更通用） ——
// 说明：保留 PointsMaterial 的使用习惯：uSize 与 pointSize 一致；uScale=绘图缓冲高度的一半
const AlphaPointsShaders = {
  vertex: /* glsl */`
    uniform float uSize;
    uniform float uScale;
    attribute vec3 color;
    attribute float alpha;
    varying vec3 vColor;
    varying float vAlpha;
    void main(){
      vColor = color;
      vAlpha = alpha;
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      float dist = max(0.0001, -mvPosition.z);
      gl_PointSize = uSize * (uScale / dist); // 等效于 PointsMaterial 的 sizeAttenuation
    }
  `,
  fragment: /* glsl */`
    precision mediump float;
    varying vec3 vColor;
    varying float vAlpha;
    void main(){
      // 画圆形点（否则默认是方块）
      vec2 uv = gl_PointCoord - 0.5;
      if(dot(uv, uv) > 0.25) discard;
      gl_FragColor = vec4(vColor, vAlpha);
    }
  `
};
function makeAlphaPointsMaterial(sizePx){
  return new THREE.ShaderMaterial({
    uniforms: {
      uSize:  { value: sizePx },
      uScale: { value: 300.0 } // 将在 resize() 时更新为绘图缓冲高度的一半
    },
    vertexShader: AlphaPointsShaders.vertex,
    fragmentShader: AlphaPointsShaders.fragment,
    transparent: true,
    depthTest: true,
    depthWrite: false,
    blending: THREE.NormalBlending
  });
}

// 简单解析：三列 PCD（x y z），忽略空行与以#开头的行
function parseXYZ(text) {
  const lines = text.split(/\r?\n/);
  let start = 0;
  let sawHeader = false;

  // 若存在 PCD 头，跳到 DATA ascii 之后
  for (let i = 0; i < lines.length; i++) {
    const L = lines[i].trim();
    if (!L) continue;
    if (/^(VERSION|FIELDS|SIZE|TYPE|COUNT|WIDTH|HEIGHT|VIEWPOINT|POINTS)\b/i.test(L)) {
      sawHeader = true;
    }
    if (/^DATA\s+ascii\b/i.test(L)) { start = i + 1; break; }
    if (/^DATA\s+/i.test(L)) { // 二进制等都不支持
      throw new Error('仅支持 PCD 文本格式：DATA ascii');
    }
  }

  const pos = [];
  for (let i = start; i < lines.length; i++) {
    const L = lines[i].trim();
    if (!L || L.startsWith('#')) continue;
    const parts = L.split(/[\s,]+/);
    if (parts.length < 3) continue;
    const x = Number(parts[0]), y = Number(parts[1]), z = Number(parts[2]);
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
      pos.push(x, y, z);
    }
  }

  // 如果曾经检测到 header，但一个点都没解析出来，给个更友好的报错
  if (sawHeader && pos.length === 0) {
    throw new Error('检测到 PCD 头，但未在 DATA ascii 段解析出坐标；请确认为 ASCII 文本而非二进制');
  }
  return new Float32Array(pos);
}

// 解析：五列 PCD（x y z sem inst）
function parseXYZL(text) {
  const lines = text.split(/\r?\n/);
  let start = 0;
  let sawHeader = false;
  let dataFormat = 'bare'; // 'bare' | 'ascii'
  let fields = null;

  // 处理 PCD 头
  for (let i = 0; i < lines.length; i++) {
    const L = lines[i].trim();
    if (!L) continue;
    if (/^(VERSION|FIELDS|SIZE|TYPE|COUNT|WIDTH|HEIGHT|VIEWPOINT|POINTS)\b/i.test(L)) {
      sawHeader = true;
    }
    if (/^FIELDS\b/i.test(L)) {
      fields = L.replace(/^FIELDS\s+/i, '').trim().split(/\s+/).map(s => s.toLowerCase());
    }
    if (/^DATA\s+ascii\b/i.test(L)) {
      start = i + 1;
      dataFormat = 'ascii';
      break;
    }
    if (/^DATA\s+/i.test(L)) {
      throw new Error('仅支持 PCD 文本格式：DATA ascii');
    }
  }

  // 列索引
  let ix = 0, iy = 1, iz = 2, isem = 3, iinst = 4;
  if (sawHeader && fields) {
    const f = fields;
    const findAny = (arr) => {
      for (const name of arr) {
        const j = f.indexOf(name);
        if (j >= 0) return j;
      }
      return -1;
    };
    ix = findAny(['x']);  iy = findAny(['y']);  iz = findAny(['z']);
    isem = findAny(['semantic','label','sem']);
    iinst = findAny(['instance','inst','id','object_id','instance_id']);

    if (ix < 0 || iy < 0 || iz < 0) throw new Error('PCD 头缺少 x/y/z 字段');
    if (isem < 0 || iinst < 0) {
      if (fields.length >= 5) { isem = (isem < 0) ? 3 : isem; iinst = (iinst < 0) ? 4 : iinst; }
      else throw new Error('未找到语义或实例字段；需要 semantic/label/sem 与 instance/inst');
    }
  } else {
    start = 0; // 无头默认前五列
  }

  const pos = [], sem = [], inst = [];
  for (let i = start; i < lines.length; i++) {
    const L = lines[i].trim();
    if (!L || L.startsWith('#')) continue;
    const parts = L.split(/[\s,]+/);

    // 这里顺便处理类似你示例中的“孤立一行 1”，长度不够会被跳过
    const need = (dataFormat === 'bare') ? 5 : Math.max(ix, iy, iz, isem, iinst) + 1;
    if (parts.length < need) continue;

    const x = Number(parts[ix]), y = Number(parts[iy]), z = Number(parts[iz]);
    const s = Number(parts[isem]), k = Number(parts[iinst]);
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) &&
        Number.isFinite(s) && Number.isFinite(k)) {
      pos.push(x, y, z);
      sem.push(s|0);
      inst.push(k|0);
    }
  }

  if (sawHeader && dataFormat !== 'ascii') throw new Error('检测到 PCD 头，但未发现 DATA ascii 段');
  if (pos.length === 0) throw new Error('未解析到任何点（需要五列 ASCII 或带 DATA ascii 的 PCD 文本）');

  return {
    positions: new Float32Array(pos),
    semantic: new Int32Array(sem),
    instance: new Int32Array(inst),
  };
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '-';
  const units = ['B','KB','MB','GB'];
  let i = 0, n = bytes;
  while (n >= 1024 && i < units.length-1) { n /= 1024; i++; }
  return `${n.toFixed(1)} ${units[i]}`;
}

function countBy(labels) {
  const map = new Map();
  for (let i = 0; i < labels.length; i++) {
    const k = labels[i];
    map.set(k, (map.get(k) || 0) + 1);
  }
  return map;
}

// 生成稳定的类别颜色（黄金角哈希）
function colorOfId(id) {
  const hue = (id * 137.508) % 360;
  const s = 65, l = 55;
  return hslToRgb(hue/360, s/100, l/100);
}
// HSL -> [r,g,b] (0..1)
function hslToRgb(h, s, l){
  if (s === 0) return [l, l, l];
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1; if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [
    hue2rgb(p, q, h + 1/3),
    hue2rgb(p, q, h),
    hue2rgb(p, q, h - 1/3),
  ];
}

/* ========== 三维视图封装 ========== */

class Viewer {
  constructor(containerEl) {
    this.el = containerEl;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf8fafc);

    const w = this.el.clientWidth || 800;
    const h = this.el.clientHeight || 500;

    this.camera = new THREE.PerspectiveCamera(60, w/h, 0.01, 1e5);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.el.appendChild(this.renderer.domElement);
    // 🔧 关键补丁：下一帧强制尺寸同步，避免初始 0×0 画布
    requestAnimationFrame(() => this.resize());

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0,0,0);

    this.pointSize = 0.15;
    this.points = null;
    this._fitBaseDist = 1; // —— 新增：用于联动时做距离比例换算 —— 

    this._animate = this._animate.bind(this);
    this._raf = requestAnimationFrame(this._animate);

    // 处理容器 resize
    const ro = new ResizeObserver(() => this.resize());
    ro.observe(this.el);
    this._ro = ro;
  }

  disposePoints() {
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
      this.points = null;
    }
  }

  loadPointCloud(positions, colors /* Float32Array[3N] 可选 */) {
    this.disposePoints();

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const hasColor = colors && colors.length === positions.length;
    if (hasColor) {
      geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    } else {
      // 默认浅灰
      const col = new Float32Array(positions.length);
      for (let i = 0; i < col.length; i+=3) { col[i]=.45; col[i+1]=.5; col[i+2]=.55; }
      geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    }

    // —— 新增：alpha 属性，默认全 1 —— 
    const N = positions.length / 3;
    const alphas = new Float32Array(N);
    alphas.fill(1.0);
    geo.setAttribute('alpha', new THREE.BufferAttribute(alphas, 1));

    // —— 使用支持透明度的点云 Shader 材质 —— 
    const mat = makeAlphaPointsMaterial(this.pointSize);
    // 同步 uScale（与 renderer 绘图缓冲高度相关）
    const db = new THREE.Vector2();
    this.renderer.getDrawingBufferSize(db);
    mat.uniforms.uScale.value = db.y * 0.5;

    const pts = new THREE.Points(geo, mat);
    this.scene.add(pts);
    this.points = pts;

    this._fitCameraToGeometry(geo);
  }

  // 根据几何体自适应相机（保持居中和合适距离）
  _fitCameraToGeometry(geo) {
    geo.computeBoundingBox();
    const box = geo.boundingBox;
    const size = new THREE.Vector3(); box.getSize(size);
    const center = new THREE.Vector3(); box.getCenter(center);

    // 把几何体平移到原点，便于重置
    geo.translate(-center.x, -center.y, -center.z);
    this.controls.target.set(0,0,0);

    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const fov = this.camera.fov * Math.PI / 180;
    let dist = (maxDim/2) / Math.tan(fov/2);
    dist *= 1.6; // 留点边距
    this._fitBaseDist = dist; // —— 新增：记住本视图加载时的基准距离 —— 
    this.camera.near = Math.max(dist/100, 0.01);
    this.camera.far = dist * 100 + maxDim*2;
    this.camera.updateProjectionMatrix();
    this.camera.position.set(0, 0, dist);
    this.controls.update();
    this.renderOnce();
  }

  setPointSize(size) {
    this.pointSize = size;
    if (this.points) {
      // this.points.material.size = size; // ← 旧 PointsMaterial 写法保留，但我们现在用 Shader
      this.points.material.uniforms.uSize.value = size;
      this.renderOnce();
    }
  }

  resetView() {
    if (this.points) {
      this._fitCameraToGeometry(this.points.geometry);
    }
  }

  setVertexColors(colors /* Float32Array[3N] */) {
    if (!this.points) return;
    const attr = this.points.geometry.getAttribute('color');
    if (!attr || attr.array.length !== colors.length) return;
    attr.array.set(colors);
    attr.needsUpdate = true;
    this.renderOnce();
  }

  // —— 新增：只更新每点透明度（0~1） ——
  setVertexAlpha(alphas /* Float32Array[N] */) {
    if (!this.points) return;
    const attr = this.points.geometry.getAttribute('alpha');
    if (!attr || attr.array.length !== alphas.length) return;
    attr.array.set(alphas);
    attr.needsUpdate = true;
    this.renderOnce();
  }

  // —— 新增：导出/应用相机状态（用于联动） ——
  getCameraState(){
    const target = this.controls.target.clone();
    const pos = this.camera.position.clone();
    const up = this.camera.up.clone();
    const dist = pos.distanceTo(target);
    return { target, pos, up, dist };
  }
  applyCameraState(state, scaleRatio=1){
    const dir = new THREE.Vector3().subVectors(state.pos, state.target);
    if (dir.lengthSq() === 0) dir.set(0,0,1);
    dir.normalize();

    // 我们的几何已被移到原点，target 固定为 (0,0,0)
    this.controls.target.set(0,0,0);
    this.camera.up.copy(state.up);

    const dist = Math.max(0.001, state.dist * scaleRatio);
    this.camera.position.copy(this.controls.target).addScaledVector(dir, dist);
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.renderOnce();
  }

  resize() {
    const w = Math.max(2, this.el.clientWidth || 0);
    const h = Math.max(2, this.el.clientHeight || 0);
    if (!w || !h) return; // 容器暂时不可见时先跳过

    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();

    // 每次 resize 同步像素比，避免模糊/拉伸
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    // 关键：同时更新 canvas 的 CSS 尺寸（不要传 false）
    this.renderer.setSize(w, h);

    // —— 同步 uScale，使点大小衰减与分辨率一致 ——
    if (this.points && this.points.material && this.points.material.uniforms) {
      const db = new THREE.Vector2();
      this.renderer.getDrawingBufferSize(db);
      this.points.material.uniforms.uScale.value = db.y * 0.5;
    }

    this.controls.update();
    this.renderOnce();
  }

  renderOnce() {
    this.renderer.render(this.scene, this.camera);
  }

  _animate() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this._raf = requestAnimationFrame(this._animate);
  }
}

/* ========== 业务逻辑 ========== */

const els = {
  pointSize: document.getElementById('pointSize'),
  resetLeft: document.getElementById('resetLeft'),
  inferBtn: document.getElementById('inferBtn'),
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  fileInfo: document.getElementById('fileInfo'),
  leftViewer: document.getElementById('leftViewer'),
  rightViewer: document.getElementById('rightViewer'),
  semanticList: document.getElementById('semanticList'),
  instanceList: document.getElementById('instanceList'),
  semanticStats: document.getElementById('semanticStats'),
  instanceStats: document.getElementById('instanceStats'),
  colorMode: document.querySelectorAll('input[name="colorMode"]'),
};

const left = new Viewer(els.leftViewer);
const right = new Viewer(els.rightViewer);

// —— 视图联动（左↔右）。如果只想“左控右”，把 SYNC_BIDIRECTIONAL 改为 false —— 
const SYNC_BIDIRECTIONAL = true;
function syncCameras(src, dst){
  if (!src || !dst) return;
  // 按加载时的基准距离做比例换算，保证两侧缩放手感一致
  const ratio = (dst._fitBaseDist && src._fitBaseDist) ? (dst._fitBaseDist / src._fitBaseDist) : 1;
  dst.applyCameraState(src.getCameraState(), ratio);
}
let _syncing = false; // 防止事件互相触发形成回环
left.controls.addEventListener('change', () => {
  if (_syncing) return; _syncing = true;
  syncCameras(left, right);
  _syncing = false;
});
if (SYNC_BIDIRECTIONAL){
  right.controls.addEventListener('change', () => {
    if (_syncing) return; _syncing = true;
    syncCameras(right, left);
    _syncing = false;
  });
}

// 左侧原始数据缓存（原文本 + positions）
let originalText = '';
let leftPositions = null;

// 右侧推理缓存
let pred = {
  positions: null,
  semantic: null,
  instance: null,
  colorsBySemantic: null, // Float32Array
  colorsByInstance: null, // Float32Array
};
let selectedSem = new Set();
let selectedInst = new Set();
let colorMode = 'semantic'; // 'semantic' | 'instance'

// 点大小联动
els.pointSize.addEventListener('input', (e) => {
  const v = Number(e.target.value);
  left.setPointSize(v);
  right.setPointSize(v);
});

// 重置按钮
els.resetLeft.addEventListener('click', () => left.resetView());

// 拖拽/点选
['dragenter','dragover'].forEach(evt =>
  els.dropzone.addEventListener(evt, (e)=>{
    e.preventDefault(); e.stopPropagation();
    els.dropzone.classList.add('dragover');
  })
);
['dragleave','drop'].forEach(evt =>
  els.dropzone.addEventListener(evt, (e)=>{
    e.preventDefault(); e.stopPropagation();
    els.dropzone.classList.remove('dragover');
  })
);
els.dropzone.addEventListener('drop', async (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) await loadLocalFile(file);
});
els.fileInput.addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) await loadLocalFile(file);
});

async function loadLocalFile(file) {
  try {
    const text = await file.text();
    const pos = parseXYZ(text);     // ← 使用兼容解析
    console.debug('[PCD] parsed points =', pos.length / 3);

    if (!pos.length) {
      els.fileInfo.innerHTML = `<div class="muted">解析失败：未发现有效的三列坐标（支持纯 xyz 或 PCD 文本 DATA ascii）</div>`;
      els.inferBtn.disabled = true;
      return;
    }

    originalText = text;
    leftPositions = pos;
    left.loadPointCloud(pos);
    requestAnimationFrame(() => {
      left.resize();
      // —— 如果右侧已有点云，则把右侧相机同步到左侧 —— 
      if (right.points) syncCameras(left, right);
    });

    els.fileInfo.innerHTML = `
      <div class="stat-row"><span>文件名</span><small>${file.name}</small></div>
      <div class="stat-row"><span>大小</span><small>${formatBytes(file.size)}</small></div>
      <div class="stat-row"><span>点数</span><small>${(pos.length/3)|0}</small></div>
    `;
    els.inferBtn.disabled = false;
  } catch (err) {
    console.error('PCD 解析异常：', err);
    els.fileInfo.innerHTML = `<div class="muted">解析异常：${err.message}</div>`;
    els.inferBtn.disabled = true;
  }
}

// 推理：把原始 PCD 文本发给后端 /infer，后端返回五列 PCD 文本
els.inferBtn.addEventListener('click', async () => {
  if (!originalText) return;
  setInferBusy(true);
  try {
    // 约定：POST JSON { pcd: "<原始文本>" }，后端返回 text/plain 的五列 PCD
    const res = await fetch('/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pcd: originalText }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    
    const data = await res.json();
    if (!data || typeof data.pcd !== 'string') {
      throw new Error('接口返回不含 pcd 字符串字段');
    }

    const parsed = parseXYZL(data.pcd);

    console.debug('[PCD-L] points=', parsed.positions.length/3, 
              'uniq_sem=', new Set(parsed.semantic).size, 
              'uniq_inst=', new Set(parsed.instance).size);
    if (!parsed.positions?.length) throw new Error('解析推理结果失败');

    pred.positions = parsed.positions;
    pred.semantic = parsed.semantic;
    pred.instance = parsed.instance;

    // 预计算两套着色
    pred.colorsBySemantic = buildColorsBy(pred.semantic);
    pred.colorsByInstance = buildColorsBy(pred.instance, /*zeroIsGray*/ true);

    // 默认按语义着色
    right.loadPointCloud(pred.positions, pred.colorsBySemantic);
    requestAnimationFrame(() => {
      right.resize();
      // —— 新加载右侧后，把右侧相机同步到左侧（视角一致） —— 
      syncCameras(left, right);
    });
    selectedSem.clear(); selectedInst.clear();
    updateFiltersAndStats();

  } catch (err) {
    alert('推理失败：' + err.message);
    console.error(err);
  } finally {
    setInferBusy(false);
  }
});

function setInferBusy(busy){
  els.inferBtn.disabled = busy || !originalText;
  els.inferBtn.textContent = busy ? '推理中...' : '推理';
}

/* ========== 筛选、统计、着色 ========== */

function buildColorsBy(intLabels, zeroIsGray=false) {
  const N = intLabels.length;
  const out = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const id = intLabels[i];
    let rgb;
    if (zeroIsGray && id === 0) {
      rgb = [0.65,0.68,0.72]; // 无实例=灰
    } else {
      rgb = colorOfId(id); // [0..1]
    }
    out[i*3] = rgb[0];
    out[i*3+1] = rgb[1];
    out[i*3+2] = rgb[2];
  }
  return out;
}

// —— 工具：唯一标签、有序 —— 
function getUniqueSorted(arr){
  return Array.from(new Set(arr)).sort((a,b)=>a-b);
}

// —— 工具：整块禁用/启用标签列表（仅禁用 checkbox，顺带做一点淡化） —— 
function setListDisabled(container, disabled){
  if (!container) return;
  container.querySelectorAll('input[type=checkbox]').forEach(cb => cb.disabled = disabled);
  container.style.opacity = disabled ? 0.6 : 1;
}

// —— 工具：整块全选/全不选 —— 
function setAllChecked(container, checked){
  if (!container) return;
  container.querySelectorAll('input[type=checkbox]').forEach(cb => cb.checked = checked);
}

// —— 根据当前着色模式重置 UI：激活栏全选，另一栏禁用；同时更新 selected 集合并应用透明度 —— 
function initializeFilterUIByMode(){
  if (!pred.semantic || !pred.instance) return;

  const semAll = getUniqueSorted(pred.semantic);
  const instAll = getUniqueSorted(pred.instance);

  if (colorMode === 'semantic') {
    selectedSem = new Set(semAll);
    selectedInst.clear();

    setAllChecked(els.semanticList, true);
    setAllChecked(els.instanceList, false);

    setListDisabled(els.semanticList, false);
    setListDisabled(els.instanceList, true);
  } else {
    selectedInst = new Set(instAll);
    selectedSem.clear();

    setAllChecked(els.instanceList, true);
    setAllChecked(els.semanticList, false);

    setListDisabled(els.semanticList, true);
    setListDisabled(els.instanceList, false);
  }
  applyHighlight();
}

function updateFiltersAndStats() {
  if (!pred.semantic || !pred.instance) {
    els.semanticList.innerHTML = '<div class="muted">无数据</div>';
    els.instanceList.innerHTML = '<div class="muted">无数据</div>';
    els.semanticStats.innerHTML = '<div class="muted">无数据</div>';
    els.instanceStats.innerHTML = '<div class="muted">无数据</div>';
    return;
  }
  // 标签列表
  const semSet = Array.from(new Set(pred.semantic)).sort((a,b)=>a-b);
  const instSet = Array.from(new Set(pred.instance)).sort((a,b)=>a-b);

  els.semanticList.innerHTML = semSet.map(id => {
    const [r,g,b] = colorOfId(id).map(x=>Math.round(x*255));
    const sw = `background: rgb(${r},${g},${b});`;
    const cid = `sem-${id}`;
    return `<div class="tag">
      <span class="swatch" style="${sw}"></span>
      <label for="${cid}"><input id="${cid}" type="checkbox" data-type="sem" data-id="${id}" /> 语义 ${id}</label>
    </div>`;
  }).join('') || '<div class="muted">无</div>';

  els.instanceList.innerHTML = instSet.map(id => {
    const rgb = (id===0) ? [166,174,184] : colorOfId(id).map(x=>Math.round(x*255));
    const sw = `background: rgb(${rgb[0]},${rgb[1]},${rgb[2]});`;
    const cid = `inst-${id}`;
    const label = (id===0) ? '（无实例）' : `实例 ${id}`;
    return `<div class="tag">
      <span class="swatch" style="${sw}"></span>
      <label for="${cid}"><input id="${cid}" type="checkbox" data-type="inst" data-id="${id}" /> ${label}</label>
    </div>`;
  }).join('') || '<div class="muted">无</div>';

  // 绑定事件
  els.semanticList.querySelectorAll('input[type=checkbox]').forEach(chk => {
    chk.addEventListener('change', onFilterChanged);
  });
  els.instanceList.querySelectorAll('input[type=checkbox]').forEach(chk => {
    chk.addEventListener('change', onFilterChanged);
  });

  // 统计
  renderStats(els.semanticStats, countBy(pred.semantic), '语义');
  renderStats(els.instanceStats, countBy(pred.instance), '实例');

  // 初次刷新颜色/透明度
  initializeFilterUIByMode();
}

function renderStats(container, map, prefix) {
  const total = Array.from(map.values()).reduce((a,b)=>a+b,0) || 1;
  const rows = Array.from(map.entries()).sort((a,b)=>a[0]-b[0]).map(([k,v])=>{
    const pct = (v*100/total).toFixed(1);
    return `<div class="stat-row"><span>${prefix} ${k}</span><small>${v} 点（${pct}%）</small></div>`;
  });
  container.innerHTML = rows.join('') || '<div class="muted">无</div>';
}

function onFilterChanged(e) {
  const id = Number(e.target.dataset.id);
  const type = e.target.dataset.type;
  if (type === 'sem') {
    if (e.target.checked) selectedSem.add(id); else selectedSem.delete(id);
  } else {
    if (e.target.checked) selectedInst.add(id); else selectedInst.delete(id);
  }
  applyHighlight();
}

// 根据 colorMode 与筛选结果更新右侧顶点透明度（未选中则变淡）
function applyHighlight() {
  if (!pred.positions) return;

  // 先确保基色与当前模式一致（当切换模式时会再次设置，这里对首次/重渲染也安全）
  const base = (colorMode === 'semantic') ? pred.colorsBySemantic : pred.colorsByInstance;
  if (base && right.points) {
    // 不强制覆盖颜色；颜色在切换模式时会刷新。这里只处理 alpha。
  }

  const N = pred.semantic ? pred.semantic.length : 0;
  if (!N) return;

  const useSem = selectedSem.size > 0;
  const useInst = selectedInst.size > 0;

  const alphas = new Float32Array(N);
  if (!useSem && !useInst) {
    alphas.fill(1.0); // 没有筛选则全部正常显示
  } else {
    const DIM = 0.12; // 未选中的透明度（越小越淡），对深/浅色背景都清晰
    for (let i = 0; i < N; i++) {
      const isSel = (useSem && selectedSem.has(pred.semantic[i])) ||
                    (useInst && selectedInst.has(pred.instance[i]));
      alphas[i] = isSel ? 1.0 : DIM;
    }
  }
  right.setVertexAlpha(alphas);
}

// 着色模式切换
els.colorMode.forEach(r => {
  r.addEventListener('change', (e) => {
    if (!e.target.checked) return;
    colorMode = e.target.value;

    // 切换模式时先刷新基色
    const base = (colorMode === 'semantic') ? pred.colorsBySemantic : pred.colorsByInstance;
    if (pred.positions && base) right.setVertexColors(base);

    // 再：激活栏全选、另一栏禁用，并应用透明度
    initializeFilterUIByMode();
  });
});
