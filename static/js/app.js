/* ══════════════════════════════════════════════════════════════
   Power Grid Topology Identifier — Dashboard Frontend Logic
   Handles API calls, chart rendering, and UI interactions
   ══════════════════════════════════════════════════════════════ */

// ── State ──────────────────────────────────────────────────────
const state = {
    currentSystem: '5',
    currentSample: 0,
    systemsInfo: {},
    lastResult: null,
};

// ── Plotly Dark Theme ──────────────────────────────────────────
const PLOTLY_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0.12)',
    font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 11 },
    margin: { l: 52, r: 16, t: 28, b: 48 },
    xaxis: {
        gridcolor:     'rgba(99,102,241,0.08)',
        zerolinecolor: 'rgba(99,102,241,0.15)',
        tickfont: { size: 10 },
    },
    yaxis: {
        gridcolor:     'rgba(99,102,241,0.08)',
        zerolinecolor: 'rgba(99,102,241,0.15)',
        tickfont: { size: 10 },
    },
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };


// ── Initialization ─────────────────────────────────────────────
async function init() {
    try {
        const res = await fetch('/api/systems');
        state.systemsInfo = await res.json();
        updateSampleMax();
        await loadSample();
    } catch (err) {
        console.error('Initialization failed:', err);
    }
}

function updateSampleMax() {
    const info = state.systemsInfo[state.currentSystem];
    if (info) {
        const el = document.getElementById('sample-index');
        el.max = info.num_samples - 1;
        el.placeholder = `0–${info.num_samples - 1}`;
    }
}


// ── API Calls ──────────────────────────────────────────────────
async function loadSample() {
    showLoading(true);
    try {
        const res = await fetch(
            `/api/sample/${state.currentSystem}/${state.currentSample}`
        );
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Request failed');
        }
        const data = await res.json();
        state.lastResult = data;
        renderAll(data);
    } catch (err) {
        console.error('Load sample failed:', err);
        showError(err.message);
    } finally {
        showLoading(false);
    }
}

async function loadRandom() {
    showLoading(true);
    try {
        const res = await fetch(`/api/random/${state.currentSystem}`);
        const data = await res.json();
        state.lastResult = data;
        state.currentSample = data.sample_index;
        document.getElementById('sample-index').value = data.sample_index;
        renderAll(data);
    } catch (err) {
        console.error('Random load failed:', err);
        showError(err.message);
    } finally {
        showLoading(false);
    }
}

async function predictCustom() {
    const textarea = document.getElementById('custom-values');
    const text = textarea.value.trim();
    if (!text) {
        showError('Please enter some values first.');
        return;
    }

    const vals = text.replace(/,/g, ' ').split(/\s+/).map(Number);
    if (vals.some(isNaN)) {
        showError('Invalid input — only numbers separated by spaces or commas.');
        return;
    }

    showLoading(true);
    try {
        const res = await fetch('/api/predict', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                values: vals,
                system_id: state.currentSystem,
            }),
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        state.lastResult = data;
        renderAll(data);
    } catch (err) {
        console.error('Custom prediction failed:', err);
        showError(err.message);
    } finally {
        showLoading(false);
    }
}


// ── Master Render ──────────────────────────────────────────────
function renderAll(data) {
    if (data.system_id && data.system_id !== state.currentSystem) {
        state.currentSystem = data.system_id;
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.system === data.system_id);
        });
        updateSampleMax();
    }
    renderMetrics(data);
    renderVoltageChart(data);
    renderCurrentChart(data);
    renderBIBCHeatmaps(data);
    renderBCBVHeatmaps(data);
    renderTopologyGraphs(data);
}


// ── Metrics ────────────────────────────────────────────────────
function renderMetrics(data) {
    const hasGroundTruth = (!data.custom_input) || (!!data.matched_ground_truth);

    // BIBC accuracy
    const accEl = document.getElementById('bibc-accuracy');
    const accCard = document.getElementById('card-bibc-acc');
    if (hasGroundTruth) {
        accEl.textContent = data.bibc_accuracy + '%';
        accEl.className = 'metric-value' +
            (data.bibc_accuracy >= 99 ? ' success-glow' : '');
        accCard.className = 'metric-card glass-card' +
            (data.bibc_accuracy >= 99 ? ' highlight-success' : '');
    } else {
        accEl.textContent = 'N/A';
        accEl.className = 'metric-value';
        accCard.className = 'metric-card glass-card';
    }

    // BCBV MAE
    const maeEl = document.getElementById('bcbv-mae');
    if (hasGroundTruth) {
        maeEl.textContent = data.bcbv_mae < 0.001
            ? data.bcbv_mae.toExponential(2) + ' Ω'
            : data.bcbv_mae.toFixed(4) + ' Ω';
    } else {
        maeEl.textContent = 'N/A';
    }

    // System info
    document.getElementById('system-info').textContent =
        data.num_buses + '-Bus';

    // Topology info
    const topoEl = document.getElementById('topo-info');
    const badgeEl = document.getElementById('topo-match-badge');
    if (hasGroundTruth) {
        topoEl.textContent = '#' + data.topology_id;
        if (data.topology_match) {
            badgeEl.className = 'match-indicator match';
            badgeEl.innerHTML = '✓ Exact Match';
        } else {
            badgeEl.className = 'match-indicator mismatch';
            badgeEl.innerHTML = '✗ Mismatch';
        }
    } else {
        topoEl.textContent = 'Custom';
        badgeEl.className = 'match-indicator';
        badgeEl.textContent = 'Live Model Input';
    }
}


// ── Voltage Chart ──────────────────────────────────────────────
function renderVoltageChart(data) {
    const buses = data.voltages.map((_, i) => `Bus ${i + 1}`);
    const trace = {
        x: buses,
        y: data.voltages,
        type: 'scatter',
        mode: 'lines+markers',
        line:   { color: '#6366f1', width: 3, shape: 'spline' },
        marker: { color: '#6366f1', size: 9,
                  line: { color: '#818cf8', width: 2 } },
        fill: 'tozeroy',
        fillcolor: 'rgba(99,102,241,0.08)',
    };

    const yMin = Math.min(...data.voltages);
    const yMax = Math.max(...data.voltages);
    const pad = (yMax - yMin) * 0.15 || 0.005;

    Plotly.newPlot('voltage-chart', [trace], {
        ...PLOTLY_LAYOUT,
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: { text: 'Bus', font: { size: 11 } } },
        yaxis: { ...PLOTLY_LAYOUT.yaxis,
                 title: { text: 'Voltage (p.u.)', font: { size: 11 } },
                 range: [yMin - pad, yMax + pad] },
    }, PLOTLY_CONFIG);
}


// ── Current Chart ──────────────────────────────────────────────
function renderCurrentChart(data) {
    const branches = data.currents.map((_, i) => `Br ${i + 1}`);
    const n = data.currents.length;

    const trace = {
        x: branches,
        y: data.currents,
        type: 'bar',
        marker: {
            color: data.currents.map((_, i) => {
                const t = i / Math.max(n - 1, 1);
                return `hsl(${185 - t * 35}, 85%, ${55 + t * 10}%)`;
            }),
            line: { color: 'rgba(6,182,212,0.4)', width: 1 },
        },
    };

    Plotly.newPlot('current-chart', [trace], {
        ...PLOTLY_LAYOUT,
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: { text: 'Branch', font: { size: 11 } } },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: 'Current Magnitude', font: { size: 11 } } },
    }, PLOTLY_CONFIG);
}


// ── BIBC Heatmaps ──────────────────────────────────────────────
function renderBIBCHeatmaps(data) {
    const isCustom = !!data.custom_input;
    const numBr = data.pred_bibc.length;
    const numNS = data.pred_bibc[0].length;

    const xLabels = Array.from({ length: numNS }, (_, i) => `Bus ${i + 2}`);
    const yLabels = Array.from({ length: numBr }, (_, i) => `Br ${i + 1}`);

    const heatLayout = {
        ...PLOTLY_LAYOUT,
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: { text: 'Non-Slack Bus', font: { size: 11 } } },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: 'Branch', font: { size: 11 } },
                 autorange: 'reversed' },
    };

    // True BIBC
    if (data.true_bibc) {
        Plotly.newPlot('true-bibc-chart', [{
            z: data.true_bibc, x: xLabels, y: yLabels,
            type: 'heatmap',
            colorscale: [[0, '#0a0a1f'], [0.5, '#312e81'], [1, '#818cf8']],
            showscale: true,
            colorbar: { title: { text: 'Value', font: { size: 10 } },
                        tickvals: [0, 1], len: 0.6 },
            hoverongaps: false,
        }], heatLayout, PLOTLY_CONFIG);
    } else {
        showPlaceholder('true-bibc-chart',
            'Live Input (No Benchmark Ground Truth)');
    }

    // Predicted BIBC
    Plotly.newPlot('pred-bibc-chart', [{
        z: data.pred_bibc, x: xLabels, y: yLabels,
        type: 'heatmap',
        colorscale: [[0, '#0a0a1f'], [0.5, '#164e63'], [1, '#22d3ee']],
        showscale: true,
        colorbar: { title: { text: 'Value', font: { size: 10 } },
                    tickvals: [0, 1], len: 0.6 },
        hoverongaps: false,
    }], heatLayout, PLOTLY_CONFIG);
}


// ── BCBV Heatmaps ──────────────────────────────────────────────
function renderBCBVHeatmaps(data) {
    const isCustom = !!data.custom_input;
    const predBcbv = data.pred_bcbv;
    const numNS = predBcbv.length;
    const numBr = predBcbv[0].length;

    const xLabels = Array.from({ length: numBr }, (_, i) => `Br ${i + 1}`);
    const yLabels = Array.from({ length: numNS }, (_, i) => `Bus ${i + 2}`);

    const heatLayout = {
        ...PLOTLY_LAYOUT,
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: { text: 'Branch', font: { size: 11 } } },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: { text: 'Non-Slack Bus', font: { size: 11 } },
                 autorange: 'reversed' },
    };

    // True BCBV
    if (data.true_bcbv) {
        const maxTrue = Math.max(...data.true_bcbv.flat().map(Math.abs)) || 0.01;
        Plotly.newPlot('true-bcbv-chart', [{
            z: data.true_bcbv, x: xLabels, y: yLabels,
            type: 'heatmap',
            colorscale: 'Magma',
            showscale: true,
            zmin: 0, zmax: maxTrue,
            colorbar: { title: { text: 'Ω', font: { size: 10 } }, len: 0.6 },
            hoverongaps: false,
        }], heatLayout, PLOTLY_CONFIG);
    } else {
        showPlaceholder('true-bcbv-chart',
            'Live Input (No Benchmark Ground Truth)');
    }

    // Predicted BCBV
    const maxPred = Math.max(...predBcbv.flat().map(Math.abs)) || 0.01;
    Plotly.newPlot('pred-bcbv-chart', [{
        z: predBcbv, x: xLabels, y: yLabels,
        type: 'heatmap',
        colorscale: 'Magma',
        showscale: true,
        zmin: 0, zmax: maxPred,
        colorbar: { title: { text: 'Ω', font: { size: 10 } }, len: 0.6 },
        hoverongaps: false,
    }], heatLayout, PLOTLY_CONFIG);
}


// ── Topology Graphs (vis-network) ──────────────────────────────
function renderTopologyGraphs(data) {
    const numBuses = data.num_buses;

    const createNodes = (color, borderColor) => {
        const arr = [];
        for (let i = 1; i <= numBuses; i++) {
            arr.push({
                id: i,
                label: String(i),
                color: {
                    background: i === 1 ? '#f59e0b' : color,
                    border:     i === 1 ? '#d97706' : borderColor,
                    highlight:  { background: '#22d3ee', border: '#06b6d4' },
                    hover:      { background: '#22d3ee', border: '#06b6d4' },
                },
                size: i === 1 ? 26 : 18,
                font: {
                    color: i === 1 ? '#000' : '#fff',
                    size: 13,
                    face: 'Inter, sans-serif',
                    bold: { color: '#fff' },
                },
                borderWidth: 2,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.35)',
                    size: 8, x: 2, y: 2,
                },
            });
        }
        return new vis.DataSet(arr);
    };

    const createEdges = (edgeList, color) =>
        new vis.DataSet(edgeList.map(([from, to], i) => ({
            id: i,
            from, to,
            color: { color, highlight: '#22d3ee', hover: '#22d3ee' },
            width: 2.5,
            arrows: { to: { enabled: true, scaleFactor: 0.65, type: 'arrow' } },
            smooth: { type: 'cubicBezier', forceDirection: 'vertical', roundness: 0.4 },
        })));

    const options = {
        layout: {
            hierarchical: {
                enabled: true,
                direction: 'UD',
                sortMethod: 'directed',
                levelSeparation: numBuses <= 5 ? 100 : 75,
                nodeSpacing: numBuses <= 5 ? 140 : 100,
                treeSpacing: 200,
            },
        },
        physics: false,
        interaction: { dragNodes: true, zoomView: true, dragView: true },
    };

    // True topology
    const trueContainer = document.getElementById('true-topology');
    if (data.true_edges && data.true_edges.length > 0) {
        trueContainer.innerHTML = '';
        new vis.Network(trueContainer, {
            nodes: createNodes('rgba(99,102,241,0.85)', '#6366f1'),
            edges: createEdges(data.true_edges, 'rgba(129,140,248,0.55)'),
        }, options);
    } else {
        trueContainer.innerHTML =
            '<div style="display:flex;align-items:center;justify-content:center;' +
            'height:100%;color:#55607a;font-size:0.9rem;font-family:Inter,sans-serif;">' +
            'Live Input (No Benchmark Ground Truth)' +
            '</div>';
    }

    // Predicted topology
    const predContainer = document.getElementById('pred-topology');
    if (data.pred_edges && data.pred_edges.length > 0) {
        predContainer.innerHTML = '';
        new vis.Network(predContainer, {
            nodes: createNodes('rgba(6,182,212,0.85)', '#06b6d4'),
            edges: createEdges(data.pred_edges, 'rgba(34,211,238,0.55)'),
        }, options);
    } else {
        predContainer.innerHTML =
            '<div style="display:flex;align-items:center;justify-content:center;' +
            'height:100%;color:#55607a;font-size:0.9rem;font-family:Inter,sans-serif;">' +
            'Could not reconstruct topology</div>';
    }
}


// ── Utility Functions ──────────────────────────────────────────
function showPlaceholder(elementId, message) {
    Plotly.newPlot(elementId, [], {
        ...PLOTLY_LAYOUT,
        xaxis: { visible: false },
        yaxis: { visible: false },
        annotations: [{
            text: message,
            showarrow: false,
            font: { size: 14, color: '#55607a', family: 'Inter, sans-serif' },
        }],
    }, PLOTLY_CONFIG);
}

function showLoading(show) {
    document.getElementById('loading-overlay')
        .classList.toggle('active', show);
}

function showError(msg) {
    // Simple alert for now; could be replaced with a toast
    alert('⚠️ ' + msg);
}

function switchSystem(systemId) {
    state.currentSystem = systemId;
    state.currentSample = 0;
    document.getElementById('sample-index').value = 0;
    updateSampleMax();

    // Update active tab styling
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.system === systemId);
    });

    loadSample();
}

function toggleCustomInput() {
    document.getElementById('custom-input-panel').classList.toggle('show');
}


// ── Event Listeners ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    init();

    // System tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchSystem(btn.dataset.system));
    });

    // Analyze button
    document.getElementById('predict-btn').addEventListener('click', () => {
        state.currentSample =
            parseInt(document.getElementById('sample-index').value) || 0;
        loadSample();
    });

    // Random button
    document.getElementById('random-btn').addEventListener('click', loadRandom);

    // Enter key on sample input
    document.getElementById('sample-index').addEventListener('keypress', e => {
        if (e.key === 'Enter') {
            state.currentSample = parseInt(e.target.value) || 0;
            loadSample();
        }
    });

    // Custom input toggle
    document.getElementById('custom-input-toggle')
        .addEventListener('click', toggleCustomInput);

    // Custom predict button
    document.getElementById('custom-predict-btn')
        .addEventListener('click', predictCustom);
});
