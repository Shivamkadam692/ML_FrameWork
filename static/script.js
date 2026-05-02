document.addEventListener('DOMContentLoaded', () => {
    const tabBtns = document.querySelectorAll('.tabs .tab-btn');
    const paramGroups = document.querySelectorAll('.param-group');
    const trainBtn = document.getElementById('trainBtn');
    const btnText = trainBtn.querySelector('.btn-text');
    const loader = trainBtn.querySelector('.loader');
    const metricBadge = document.getElementById('metricBadge');
    const emptyState = document.getElementById('emptyState');
    const errorMsg = document.getElementById('errorMsg');
    const datasetDesc = document.getElementById('datasetDesc');
    const chartCanvas = document.getElementById('lossChart');
    const perfCanvas = document.getElementById('perfChart');
    const datasetSelect = document.getElementById('datasetSelect');
    const viewLossBtn = document.getElementById('viewLossBtn');
    const viewPerfBtn = document.getElementById('viewPerfBtn');
    const viewDiagBtn = document.getElementById('viewDiagBtn');
    const diagContainer = document.getElementById('diagnosticsContainer');
    const featureCanvas = document.getElementById('featureImportanceChart');
    const residualCanvas = document.getElementById('residualsChart');
    const cmContainer = document.getElementById('confusionMatrixContainer');
    const metricsCards = document.getElementById('metricsCards');

    let lossChart = null, perfChart = null, featureChart = null, residualChart = null;
    let currentModel = 'linear';
    let currentData = null;

    const descriptions = {
        'linear': 'California Housing Dataset - A regression task predicting median house values based on 8 features.',
        'housing': 'California Housing Dataset - A regression task predicting median house values based on 8 features.',
        'logistic': 'Breast Cancer Dataset - A classification task predicting whether a tumor is malignant or benign.',
        'breast_cancer': 'Breast Cancer Dataset - A classification task predicting whether a tumor is malignant or benign.',
        'tree': 'Iris Dataset - A multi-class classification task identifying iris flower species.',
        'iris': 'Iris Dataset - A multi-class classification task identifying iris flower species.'
    };

    const howItWorks = {
        'linear': 'Calculates a weighted sum of the input features plus a bias term. It uses <strong>Gradient Descent</strong> to iteratively decrease the <strong>Mean Squared Error (MSE)</strong> by updating weights based on their gradients.',
        'logistic': 'Passes a weighted sum of inputs through a <strong>Sigmoid function</strong> to output a probability (0-1). It uses Gradient Descent to minimize the <strong>Binary Cross-Entropy Loss</strong>, finding the best boundary between classes.',
        'tree': 'Builds a tree by recursively partitioning data. At each step, it finds the feature and split point that maximizes information gain using <strong>Entropy</strong>. It stops when reaching max depth or minimum samples.'
    };

    // Model Tab Switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentModel = btn.dataset.model;

            paramGroups.forEach(g => g.classList.remove('active'));
            if (currentModel === 'tree') {
                document.getElementById('tree-params').classList.add('active');
            } else {
                document.getElementById('gd-params').classList.add('active');
                const lrInput = document.getElementById('learningRate');
                lrInput.value = currentModel === 'logistic' ? '0.05' : '0.01';
            }

            Array.from(datasetSelect.options).forEach(opt => {
                if (currentModel === 'linear') {
                    opt.disabled = opt.value !== 'housing';
                    if (opt.value === 'housing') opt.selected = true;
                } else {
                    opt.disabled = opt.value === 'housing';
                    if (!opt.disabled && !document.querySelector('#datasetSelect option:checked:not(:disabled)')) opt.selected = true;
                }
            });
            if (currentModel === 'logistic') datasetSelect.value = 'breast_cancer';
            if (currentModel === 'tree') datasetSelect.value = 'iris';

            datasetDesc.textContent = descriptions[datasetSelect.value] || '';
            document.getElementById('howItWorksDesc').innerHTML = howItWorks[currentModel];
        });
    });

    datasetSelect.addEventListener('change', () => {
        datasetDesc.textContent = descriptions[datasetSelect.value] || '';
    });

    // View Switching
    function switchView(view) {
        viewLossBtn.classList.remove('active');
        viewPerfBtn.classList.remove('active');
        viewDiagBtn.classList.remove('active');
        chartCanvas.style.display = 'none';
        perfCanvas.style.display = 'none';
        diagContainer.style.display = 'none';
        emptyState.classList.add('hidden');

        if (view === 'loss') {
            viewLossBtn.classList.add('active');
            if (!currentData) { showEmpty('Train a model to see the Learning Curve', 'A Learning Curve shows how the model\'s error (Loss) decreases over time.'); return; }
            if (currentModel === 'tree') { showEmpty('Tree Constructed Successfully!', 'Decision Trees don\'t use gradient descent, so there is no loss curve. Check Predictions or Model Diagnostics!'); return; }
            chartCanvas.style.display = 'block';
        } else if (view === 'perf') {
            viewPerfBtn.classList.add('active');
            if (!currentData) { showEmpty('Train a model to see Performance', 'This view shows how close the model\'s predictions are to the actual values.'); return; }
            perfCanvas.style.display = 'block';
        } else if (view === 'diag') {
            viewDiagBtn.classList.add('active');
            if (!currentData) { showEmpty('Train a model to see Diagnostics', 'Feature importance, residuals analysis, and confusion matrices help you understand what the model learned.'); return; }
            diagContainer.style.display = 'flex';
        }
    }

    function showEmpty(title, desc) {
        emptyState.classList.remove('hidden');
        emptyState.innerHTML = `
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.8;margin-bottom:0.5rem;">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                <line x1="12" y1="22.08" x2="12" y2="12"></line>
            </svg>
            <div style="text-align:center;max-width:80%;">
                <p style="margin-bottom:0.5rem;color:var(--text-primary);font-weight:500;">${title}</p>
                <span class="help-text" style="font-size:0.85rem;color:var(--text-secondary);">${desc}</span>
            </div>`;
    }

    viewLossBtn.addEventListener('click', () => switchView('loss'));
    viewPerfBtn.addEventListener('click', () => switchView('perf'));
    viewDiagBtn.addEventListener('click', () => switchView('diag'));

    // Train
    trainBtn.addEventListener('click', async () => {
        let payload = {};
        if (currentModel === 'tree') {
            payload = { maxDepth: parseInt(document.getElementById('maxDepth').value), minSamples: parseInt(document.getElementById('minSamples').value), dataset: datasetSelect.value };
        } else {
            payload = { learningRate: parseFloat(document.getElementById('learningRate').value), iterations: parseInt(document.getElementById('iterations').value), dataset: datasetSelect.value };
        }

        trainBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        errorMsg.classList.add('hidden');
        metricBadge.classList.add('hidden');
        metricsCards.classList.add('hidden');

        try {
            const response = await fetch(`/api/train/${currentModel}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await response.json();

            if (data.success) {
                metricBadge.textContent = data.metric;
                metricBadge.classList.remove('hidden');
                emptyState.classList.add('hidden');
                currentData = data;

                plotLoss(data.loss_history, currentModel);
                plotPerformance(data.y_test, data.predictions, datasetSelect.value);
                plotDiagnostics(data, currentModel);
                updateMetricsCards(data, currentModel);

                if (currentModel === 'tree') { switchView('perf'); } else { switchView('loss'); }
            } else {
                showError(data.error);
            }
        } catch (err) {
            showError("Failed to connect to the backend server. Is Flask running?");
            console.error(err);
        } finally {
            trainBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });

    function showError(msg) { errorMsg.textContent = msg; errorMsg.classList.remove('hidden'); }

    // Chart defaults
    Chart.defaults.color = '#8b949e';
    Chart.defaults.font.family = 'Outfit';

    function chartTooltip() {
        return { mode: 'index', intersect: false, backgroundColor: 'rgba(22,27,34,0.9)', titleColor: '#e6edf3', bodyColor: '#e6edf3', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1 };
    }

    // Loss Chart
    function plotLoss(lossHistory, modelType) {
        if (lossChart) lossChart.destroy();
        if (modelType === 'tree' || !lossHistory || lossHistory.length === 0) return;

        const maxPts = 100;
        let labels = Array.from({length: lossHistory.length}, (_, i) => i + 1);
        let plotData = lossHistory;
        if (lossHistory.length > maxPts) {
            const step = Math.floor(lossHistory.length / maxPts);
            labels = labels.filter((_, i) => i % step === 0);
            plotData = lossHistory.filter((_, i) => i % step === 0);
        }

        lossChart = new Chart(chartCanvas.getContext('2d'), {
            type: 'line',
            data: { labels, datasets: [{ label: 'Training Loss', data: plotData, borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)', borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, fill: true, tension: 0.4 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: { display: true, text: 'Learning Curve: How the model\'s error decreases over time.', color: 'rgba(230,237,243,0.8)', font: { size: 14, weight: 'normal', family: 'Outfit' }, padding: { top: 0, bottom: 20 } }, tooltip: chartTooltip() },
                scales: { x: { grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Iteration', color: '#8b949e', font: { size: 13 } } }, y: { grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Loss', color: '#8b949e', font: { size: 13 } } } },
                interaction: { mode: 'nearest', axis: 'x', intersect: false }
            }
        });
    }

    // Performance Chart
    function plotPerformance(yTest, predictions, dataset) {
        if (perfChart) perfChart.destroy();
        const labels = Array.from({length: yTest.length}, (_, i) => i + 1);

        perfChart = new Chart(perfCanvas.getContext('2d'), {
            type: 'line',
            data: { labels, datasets: [
                { label: 'Actual Values', data: yTest, borderColor: '#3fb950', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#3fb950', showLine: false },
                { label: 'Predicted Values', data: predictions, borderColor: '#f85149', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 3, pointStyle: 'crossRot', pointBackgroundColor: '#f85149', showLine: false }
            ] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: true, labels: { color: '#8b949e', font: { family: 'Outfit' } } }, title: { display: true, text: dataset === 'housing' ? 'Predictions vs Actual (Subset)' : 'Classifications vs Actual (Subset)', color: 'rgba(230,237,243,0.8)', font: { size: 14, weight: 'normal', family: 'Outfit' }, padding: { top: 0, bottom: 20 } }, tooltip: chartTooltip() },
                scales: { x: { grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Sample Index', color: '#8b949e', font: { size: 13 } } }, y: { grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: dataset === 'housing' ? 'Value' : 'Class Label', color: '#8b949e', font: { size: 13 } }, ticks: dataset !== 'housing' ? { stepSize: 1 } : {} } }
            }
        });
    }

    // ─── Model Diagnostics ───
    function plotDiagnostics(data, modelType) {
        if (featureChart) featureChart.destroy();
        if (residualChart) residualChart.destroy();
        cmContainer.innerHTML = '';
        residualCanvas.style.display = 'none';
        featureCanvas.style.display = 'none';

        // 1) Feature Importance / Weights
        if (data.weights && data.weights.length > 0) {
            featureCanvas.style.display = 'block';
            const labels = data.feature_names && data.feature_names.length > 0
                ? data.feature_names
                : Array.from({length: data.weights.length}, (_, i) => `Feature ${i+1}`);

            let paired = labels.map((l, i) => ({ label: l, value: data.weights[i] }));
            paired.sort((a, b) => b.value - a.value);
            if (paired.length > 12) paired = paired.slice(0, 12);

            const title = modelType === 'tree' ? 'Feature Importance (Split Frequency)' : 'Feature Importance (|Learned Weights|)';
            const colors = paired.map((_, i) => `hsla(${210 + i * 15}, 80%, 65%, 0.6)`);
            const borders = paired.map((_, i) => `hsla(${210 + i * 15}, 80%, 65%, 1)`);

            featureChart = new Chart(featureCanvas.getContext('2d'), {
                type: 'bar',
                data: { labels: paired.map(p => p.label), datasets: [{ label: 'Importance', data: paired.map(p => p.value), backgroundColor: colors, borderColor: borders, borderWidth: 1, borderRadius: 4 }] },
                options: {
                    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false }, title: { display: true, text: title, color: 'rgba(230,237,243,0.8)', font: { size: 13, weight: 'normal', family: 'Outfit' }, padding: { bottom: 12 } } },
                    scales: { x: { grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: modelType === 'tree' ? 'Relative Importance' : 'Absolute Weight', color: '#8b949e' } }, y: { grid: { display: false }, ticks: { color: '#8b949e', font: { size: 11 } } } }
                }
            });
        }

        // 2) Residuals Plot (Linear Regression only)
        if (modelType === 'linear' && data.residuals) {
            residualCanvas.style.display = 'block';
            residualChart = new Chart(residualCanvas.getContext('2d'), {
                type: 'scatter',
                data: { datasets: [{
                    label: 'Residuals',
                    data: data.residuals.map((r, i) => ({ x: data.predictions[i], y: r })),
                    backgroundColor: 'rgba(240,180,41,0.5)',
                    borderColor: 'rgba(240,180,41,0.8)',
                    pointRadius: 4, pointHoverRadius: 6
                }, {
                    label: 'Zero Line',
                    data: [{ x: Math.min(...data.predictions), y: 0 }, { x: Math.max(...data.predictions), y: 0 }],
                    type: 'line', borderColor: 'rgba(248,81,73,0.5)', borderWidth: 2, borderDash: [6, 4], pointRadius: 0, fill: false
                }] },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: true, labels: { color: '#8b949e' } }, title: { display: true, text: 'Residuals vs Predicted — Points near zero = good fit', color: 'rgba(230,237,243,0.8)', font: { size: 13, weight: 'normal', family: 'Outfit' }, padding: { bottom: 12 } }, tooltip: chartTooltip() },
                    scales: { x: { grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Predicted Value', color: '#8b949e' } }, y: { grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Residual (Actual − Predicted)', color: '#8b949e' } } }
                }
            });
        }

        // 3) Confusion Matrix (Classification only)
        if ((modelType === 'logistic' || modelType === 'tree') && data.confusion_matrix && data.classes) {
            const cm = data.confusion_matrix;
            const maxVal = Math.max(...cm.flat());

            let html = `<div style="background:rgba(22,27,34,0.6);border:1px solid rgba(255,255,255,0.1);border-radius:10px;padding:1.2rem;text-align:center;width:100%;margin-top:0.5rem;">
                <h3 style="margin-bottom:0.8rem;color:rgba(230,237,243,0.9);font-size:13px;font-weight:normal;font-family:'Outfit';">Confusion Matrix — Correct predictions on diagonal (green), errors off-diagonal (red)</h3>
                <table style="border-collapse:collapse;margin:0 auto;color:#e6edf3;font-family:'Outfit';">
                    <tr><td style="padding:6px;"></td>`;

            data.classes.forEach(c => { html += `<td style="padding:8px 12px;font-weight:600;color:#8b949e;border-bottom:1px solid rgba(255,255,255,0.1);font-size:0.8rem;">Pred ${c}</td>`; });
            html += `</tr>`;

            cm.forEach((row, i) => {
                html += `<tr><td style="padding:8px 12px;font-weight:600;color:#8b949e;border-right:1px solid rgba(255,255,255,0.1);text-align:right;font-size:0.8rem;">True ${data.classes[i]}</td>`;
                row.forEach((val, j) => {
                    const intensity = maxVal > 0 ? val / maxVal : 0;
                    const bg = i === j
                        ? `rgba(63,185,80,${0.1 + intensity * 0.45})`
                        : (val > 0 ? `rgba(248,81,73,${0.1 + intensity * 0.35})` : 'rgba(255,255,255,0.02)');
                    html += `<td style="padding:12px 16px;font-size:1.05rem;font-weight:600;background:${bg};border:1px solid rgba(255,255,255,0.05);">${val}</td>`;
                });
                html += `</tr>`;
            });
            html += `</table></div>`;
            cmContainer.innerHTML = html;
        }
    }


    // Metrics Cards
    function updateMetricsCards(data, modelType) {
        metricsCards.classList.remove('hidden');

        if (modelType === 'linear') {
            setCard(1, 'MSE', data.mse);
            setCard(2, 'R² Score', data.r2_score);
            setCard(3, 'Train Samples', data.train_samples);
            setCard(4, 'Features', data.n_features);
        } else {
            setCard(1, 'Accuracy', (data.accuracy * 100).toFixed(1) + '%');
            setCard(2, 'Precision', (data.precision * 100).toFixed(1) + '%');
            setCard(3, 'Recall', (data.recall * 100).toFixed(1) + '%');
            setCard(4, 'F1 Score', (data.f1_score * 100).toFixed(1) + '%');
        }
    }

    function setCard(n, label, value) {
        document.getElementById(`metricLabel${n}`).textContent = label;
        const valEl = document.getElementById(`metricValue${n}`);
        valEl.textContent = value;
        valEl.classList.remove('animate');
        void valEl.offsetWidth; // trigger reflow
        valEl.classList.add('animate');
    }
});
