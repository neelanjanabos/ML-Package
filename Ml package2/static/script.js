/**
 * JavaScript interactions for Network Attack Severity Prediction System.
 * Handles Chart.js visualization rendering, preset population, and tab switching.
 */

document.addEventListener('DOMContentLoaded', function () {
    // --------------------------------------------------------------------------
    // 1. Prediction Form Preset Loader
    // --------------------------------------------------------------------------
    const presetButtons = document.querySelectorAll('.btn-preset');
    if (presetButtons.length > 0) {
        presetButtons.forEach(btn => {
            btn.addEventListener('click', function () {
                const presetKey = this.getAttribute('data-preset');
                loadPreset(presetKey);
            });
        });
    }

    // --------------------------------------------------------------------------
    // 2. Confusion Matrix Tab Switcher (on Models page)
    // --------------------------------------------------------------------------
    const cmTabs = document.querySelectorAll('.cm-tab-btn');
    if (cmTabs.length > 0) {
        cmTabs.forEach(tab => {
            tab.addEventListener('click', function () {
                const targetId = this.getAttribute('data-target');
                
                // Toggle active button
                cmTabs.forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                
                // Toggle active matrix card
                document.querySelectorAll('.cm-matrix-card').forEach(card => {
                    card.style.display = 'none';
                });
                const activeCard = document.getElementById(targetId);
                if (activeCard) {
                    activeCard.style.display = 'block';
                }
            });
        });
    }
});

/**
 * Loads pre-defined network packet parameters into the prediction form.
 */
function loadPreset(presetKey) {
    fetch('/api/presets')
        .then(response => response.json())
        .then(presets => {
            if (presets && presets[presetKey]) {
                const values = presets[presetKey].values;
                for (const [key, val] of Object.entries(values)) {
                    const input = document.getElementById('input_' + key);
                    if (input) {
                        input.value = val;
                        // Trigger visual feedback highlight
                        input.style.backgroundColor = '#eff6ff';
                        setTimeout(() => {
                            input.style.backgroundColor = '#ffffff';
                        }, 400);
                    }
                }
            }
        })
        .catch(err => console.error('Error fetching presets:', err));
}

// ------------------------------------------------------------------------------
// CHART HELPERS
// ------------------------------------------------------------------------------

const SEVERITY_COLORS = {
    'LOW': '#10b981',
    'MEDIUM': '#f59e0b',
    'HIGH': '#f97316',
    'CRITICAL': '#ef4444'
};

/**
 * Renders home page distribution charts
 */
function initHomeCharts(severityData, attackData) {
    // 1. Severity Doughnut Chart
    const ctxSev = document.getElementById('severityChart');
    if (ctxSev) {
        const labels = Object.keys(severityData);
        const data = Object.values(severityData);
        const bgColors = labels.map(label => SEVERITY_COLORS[label] || '#94a3b8');

        new Chart(ctxSev, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: bgColors,
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }

    // 2. Attack Category Horizontal Bar Chart
    const ctxAtk = document.getElementById('attackChart');
    if (ctxAtk) {
        const labels = Object.keys(attackData);
        const data = Object.values(attackData);

        new Chart(ctxAtk, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Packet Flow Count',
                    data: data,
                    backgroundColor: '#0ea5e9',
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { beginAtZero: true }
                }
            }
        });
    }
}

/**
 * Renders model comparison bar charts
 */
function initModelsCharts(modelsData) {
    const modelNames = Object.keys(modelsData);
    const accuracies = modelNames.map(m => modelsData[m].accuracy);
    const f1Scores = modelNames.map(m => modelsData[m].f1_score);

    // Accuracy Chart
    const ctxAcc = document.getElementById('accuracyChart');
    if (ctxAcc) {
        new Chart(ctxAcc, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [{
                    label: 'Accuracy (%)',
                    data: accuracies,
                    backgroundColor: '#2563eb',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 40, max: 100 }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // F1-Score Chart
    const ctxF1 = document.getElementById('f1Chart');
    if (ctxF1) {
        new Chart(ctxF1, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [{
                    label: 'F1-Score (%)',
                    data: f1Scores,
                    backgroundColor: '#10b981',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 40, max: 100 }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
}

/**
 * Renders batch upload summary chart
 */
function initUploadCharts(severityDist) {
    const ctx = document.getElementById('batchSeverityChart');
    if (ctx && severityDist) {
        const labels = Object.keys(severityDist);
        const data = Object.values(severityDist);
        const colors = labels.map(l => SEVERITY_COLORS[l] || '#94a3b8');

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }
}
