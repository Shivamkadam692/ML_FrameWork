document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
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
    const viewBackendBtn = document.getElementById('viewBackendBtn');
    const backendCanvas = document.getElementById('backendChart');
    const cmContainer = document.getElementById('confusionMatrixContainer');
    
    // Chart Instances
    let lossChart = null;
    let perfChart = null;
    let backendChart = null;
    let currentModel = 'linear';
    let currentData = null;

    // Model Descriptions
    const descriptions = {
        'linear': 'California Housing Dataset - A regression task predicting median house values based on 8 features.',
        'logistic': 'Breast Cancer Dataset - A classification task predicting whether a tumor is malignant or benign.',
        'tree': 'Iris Dataset - A multi-class classification task identifying iris flower species based on sepal/petal dimensions.'
    };

    const howItWorks = {
        'linear': 'Calculates a weighted sum of the input features plus a bias term. It uses <strong>Gradient Descent</strong> to iteratively decrease the <strong>Mean Squared Error (MSE)</strong> by updating weights based on their gradients.',
        'logistic': 'Passes a weighted sum of inputs through a <strong>Sigmoid function</strong> to output a probability (0-1). It uses Gradient Descent to minimize the <strong>Binary Cross-Entropy Loss</strong>, finding the best boundary between classes.',
        'tree': 'Builds a tree by recursively partitioning data. At each step, it finds the feature and split point that maximizes information gain using <strong>Gini Impurity</strong>. It stops when reaching max depth or minimum samples.'
    };

    // Tab Switching Logic
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active styling
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            currentModel = btn.dataset.model;

            // Toggle parameter visibility
            paramGroups.forEach(group => group.classList.remove('active'));
            if (currentModel === 'tree') {
                document.getElementById('tree-params').classList.add('active');
            } else {
                document.getElementById('gd-params').classList.add('active');
                
                // Set default LR for specific models
                const lrInput = document.getElementById('learningRate');
                if (currentModel === 'linear') lrInput.value = '0.01';
                if (currentModel === 'logistic') lrInput.value = '0.05';
            }

            // Update Dataset Options
            Array.from(datasetSelect.options).forEach(opt => {
                if (currentModel === 'linear') {
                    if (opt.value === 'housing') { opt.disabled = false; opt.selected = true; }
                    else opt.disabled = true;
                } else {
                    if (opt.value === 'housing') opt.disabled = true;
                    else { opt.disabled = false; if (opt.value === 'breast_cancer') opt.selected = true; }
                }
            });

            // Update description
            datasetDesc.textContent = descriptions[datasetSelect.value];
            document.getElementById('howItWorksDesc').innerHTML = howItWorks[currentModel];
        });
    });

    datasetSelect.addEventListener('change', () => {
        datasetDesc.textContent = descriptions[datasetSelect.value];
    });

    function switchView(view) {
        viewLossBtn.classList.remove('active');
        viewPerfBtn.classList.remove('active');
        viewBackendBtn.classList.remove('active');
        chartCanvas.style.display = 'none';
        perfCanvas.style.display = 'none';
        backendCanvas.style.display = 'none';
        cmContainer.style.display = 'none';
        emptyState.classList.add('hidden');

        if (view === 'loss') {
            viewLossBtn.classList.add('active');
            chartCanvas.style.display = 'block';
            if (currentModel === 'tree' || (!lossChart && currentModel !== 'tree')) {
                emptyState.classList.remove('hidden');
                emptyState.innerHTML = `
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.8; margin-bottom: 0.5rem;"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
                    <div style="text-align: center; max-width: 80%;">
                        <p style="margin-bottom: 0.5rem; color: var(--text-primary); font-weight: 500;">Tree Constructed Successfully!</p>
                        <span class="help-text" style="font-size: 0.85rem; color: var(--text-secondary);">Decision Trees don't use gradient descent, so there is no continuous loss curve to visualize. Check out the Predictions or Backend Insights!</span>
                    </div>
                `;
                if (!currentData) {
                    emptyState.innerHTML = `
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
                        <div style="text-align: center; max-width: 80%;">
                            <p style="margin-bottom: 0.5rem; color: var(--text-primary); font-weight: 500;">Train a model to see the Learning Curve</p>
                            <span class="help-text" style="font-size: 0.85rem;">A Learning Curve shows how the model's error (Loss) decreases over time.</span>
                        </div>
                    `;
                }
            }
        } else if (view === 'perf') {
            viewPerfBtn.classList.add('active');
            perfCanvas.style.display = 'block';
            if (!currentData) {
                emptyState.classList.remove('hidden');
                emptyState.innerHTML = `
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.8; margin-bottom: 0.5rem;"><path d="M12 20v-6M6 20V10M18 20V4"></path></svg>
                    <div style="text-align: center; max-width: 80%;">
                        <p style="margin-bottom: 0.5rem; color: var(--text-primary); font-weight: 500;">Train a model to see Performance</p>
                        <span class="help-text" style="font-size: 0.85rem; color: var(--text-secondary);">This view shows how close the model's predictions are to the actual values.</span>
                    </div>
                `;
            }
        } else if (view === 'backend') {
            viewBackendBtn.classList.add('active');
            if (!currentData) {
                emptyState.classList.remove('hidden');
                emptyState.innerHTML = `
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.8; margin-bottom: 0.5rem;"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
                    <div style="text-align: center; max-width: 80%;">
                        <p style="margin-bottom: 0.5rem; color: var(--text-primary); font-weight: 500;">Train a model to see Backend Insights</p>
                        <span class="help-text" style="font-size: 0.85rem; color: var(--text-secondary);">This view peeks into the model's inner workings like learned weights or confusion matrices.</span>
                    </div>
                `;
            } else {
                if (currentModel === 'linear') {
                    backendCanvas.style.display = 'block';
                    backendCanvas.style.height = '100%';
                } else if (currentModel === 'logistic') {
                    backendCanvas.style.display = 'block';
                    backendCanvas.style.height = '60%';
                    cmContainer.style.display = 'flex';
                    cmContainer.style.height = '40%';
                } else if (currentModel === 'tree') {
                    cmContainer.style.display = 'flex';
                    cmContainer.style.height = '100%';
                }
            }
        }
    }

    viewLossBtn.addEventListener('click', () => switchView('loss'));
    viewPerfBtn.addEventListener('click', () => switchView('perf'));
    viewBackendBtn.addEventListener('click', () => switchView('backend'));

    // Train Model Logic
    trainBtn.addEventListener('click', async () => {
        // Build Payload
        let payload = {};
        if (currentModel === 'tree') {
            payload = {
                maxDepth: parseInt(document.getElementById('maxDepth').value),
                minSamples: parseInt(document.getElementById('minSamples').value),
                dataset: datasetSelect.value
            };
        } else {
            payload = {
                learningRate: parseFloat(document.getElementById('learningRate').value),
                iterations: parseInt(document.getElementById('iterations').value),
                dataset: datasetSelect.value
            };
        }

        // UI State: Loading
        trainBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        errorMsg.classList.add('hidden');
        metricBadge.classList.add('hidden');

        try {
            const response = await fetch(`/api/train/${currentModel}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (data.success) {
                // Display Metric
                metricBadge.textContent = data.metric;
                metricBadge.classList.remove('hidden');
                
                // Hide empty state
                emptyState.classList.add('hidden');

                currentData = data;

                // Plot Loss Curve
                plotLoss(data.loss_history, currentModel);
                
                // Plot Performance Curve
                plotPerformance(data.y_test, data.predictions, datasetSelect.value);

                // Plot Backend Insights
                plotBackendInsights(data, currentModel);
                
                // Switch to Performance view if Tree is selected (since tree has no loss)
                if (currentModel === 'tree') {
                    viewPerfBtn.click();
                } else {
                    viewLossBtn.click();
                }
            } else {
                showError(data.error);
            }
        } catch (err) {
            showError("Failed to connect to the backend server. Is Flask running?");
            console.error(err);
        } finally {
            // UI State: Reset
            trainBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove('hidden');
    }

    function plotLoss(lossHistory, modelType) {
        if (lossChart) {
            lossChart.destroy();
        }

        if (modelType === 'tree' || !lossHistory || lossHistory.length === 0) {
            // Decision trees don't use GD, so no loss curve
            emptyState.innerHTML = `
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.8; margin-bottom: 0.5rem;"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
                <div style="text-align: center; max-width: 80%;">
                    <p style="margin-bottom: 0.5rem; color: var(--text-primary); font-weight: 500;">Tree Constructed Successfully!</p>
                    <span class="help-text" style="font-size: 0.85rem; color: var(--text-secondary);">Decision Trees learn by making direct mathematical splits in the dataset all at once. Unlike gradient descent, they do not incrementally update weights. Therefore, there is no continuous error curve to visualize.</span>
                </div>
            `;
            emptyState.classList.remove('hidden');
            return;
        }

        const ctx = chartCanvas.getContext('2d');
        
        // Downsample for rendering performance if iterations are high
        const maxPoints = 100;
        let labels = Array.from({length: lossHistory.length}, (_, i) => i + 1);
        let plotData = lossHistory;

        if (lossHistory.length > maxPoints) {
            const step = Math.floor(lossHistory.length / maxPoints);
            labels = labels.filter((_, i) => i % step === 0);
            plotData = lossHistory.filter((_, i) => i % step === 0);
        }

        Chart.defaults.color = '#8b949e';
        Chart.defaults.font.family = 'Outfit';

        lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Training Loss',
                    data: plotData,
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: 'Learning Curve: How the model\'s error decreases over time.',
                        color: 'rgba(230, 237, 243, 0.8)',
                        font: { size: 14, weight: 'normal', family: 'Outfit' },
                        padding: { top: 0, bottom: 20 }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(22, 27, 34, 0.9)',
                        titleColor: '#e6edf3',
                        bodyColor: '#e6edf3',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: 'Iteration (Training Step)', color: '#8b949e', font: { size: 13 } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: 'Loss (Amount of Error)', color: '#8b949e', font: { size: 13 } }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    function plotPerformance(yTest, predictions, dataset) {
        if (perfChart) {
            perfChart.destroy();
        }

        const ctx = perfCanvas.getContext('2d');
        const labels = Array.from({length: yTest.length}, (_, i) => i + 1);

        perfChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Actual Values',
                        data: yTest,
                        borderColor: '#3fb950', // Green
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 3,
                        pointBackgroundColor: '#3fb950',
                        showLine: false // Scatter plot style
                    },
                    {
                        label: 'Predicted Values',
                        data: predictions,
                        borderColor: '#f85149', // Red
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 3,
                        pointStyle: 'crossRot',
                        pointBackgroundColor: '#f85149',
                        showLine: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#8b949e', font: { family: 'Outfit' } }
                    },
                    title: {
                        display: true,
                        text: dataset === 'housing' ? 'Predictions vs Actual (Subset)' : 'Classifications vs Actual (Subset)',
                        color: 'rgba(230, 237, 243, 0.8)',
                        font: { size: 14, weight: 'normal', family: 'Outfit' },
                        padding: { top: 0, bottom: 20 }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(22, 27, 34, 0.9)',
                        titleColor: '#e6edf3',
                        bodyColor: '#e6edf3',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: 'Sample Index', color: '#8b949e', font: { size: 13 } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: dataset === 'housing' ? 'Value' : 'Class Label', color: '#8b949e', font: { size: 13 } },
                        ticks: dataset !== 'housing' ? { stepSize: 1 } : {}
                    }
                }
            }
        });
    }

    function plotBackendInsights(data, modelType) {
        if (backendChart) {
            backendChart.destroy();
        }
        cmContainer.innerHTML = ''; 
        
        if (modelType === 'linear' || modelType === 'logistic') {
            if (data.weights) {
                const ctx = backendCanvas.getContext('2d');
                let labels = data.feature_names && data.feature_names.length > 0 ? 
                             data.feature_names : 
                             Array.from({length: data.weights.length}, (_, i) => `Feature ${i+1}`);
                
                backendChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Absolute Feature Weight',
                            data: data.weights,
                            backgroundColor: 'rgba(88, 166, 255, 0.4)',
                            borderColor: '#58a6ff',
                            borderWidth: 1,
                            borderRadius: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            title: {
                                display: true,
                                text: 'Feature Importance (Magnitude of Learned Weights)',
                                color: 'rgba(230, 237, 243, 0.8)',
                                font: { size: 14, weight: 'normal', family: 'Outfit' },
                                padding: { top: 0, bottom: 20 }
                            }
                        },
                        scales: {
                            y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, title: { display: true, text: 'Absolute Weight', color: '#8b949e' } },
                            x: { grid: { display: false }, ticks: { color: '#8b949e', maxRotation: 45, minRotation: 0 } }
                        }
                    }
                });
            }
        }

        if (modelType === 'logistic' || modelType === 'tree') {
            if (data.confusion_matrix && data.classes) {
                let tableHtml = `<div style="background: rgba(22, 27, 34, 0.6); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 1.5rem; text-align: center; margin: 0 auto;">
                    <h3 style="margin-bottom: 1rem; color: rgba(230, 237, 243, 0.9); font-size: 15px; font-weight: normal; font-family: 'Outfit';">Confusion Matrix</h3>
                    <table style="border-collapse: collapse; margin: 0 auto; color: #e6edf3; font-family: 'Outfit';">
                        <tr>
                            <td style="padding: 8px;"></td>`;
                
                // Headers
                data.classes.forEach(c => {
                    tableHtml += `<td style="padding: 8px; font-weight: 600; color: #8b949e; border-bottom: 1px solid rgba(255,255,255,0.1);">Pred ${c}</td>`;
                });
                tableHtml += `</tr>`;

                // Rows
                data.confusion_matrix.forEach((row, i) => {
                    tableHtml += `<tr><td style="padding: 8px; font-weight: 600; color: #8b949e; border-right: 1px solid rgba(255,255,255,0.1); text-align: right;">Actual ${data.classes[i]}</td>`;
                    row.forEach((val, j) => {
                        const color = i === j ? 'rgba(63, 185, 80, 0.2)' : (val > 0 ? 'rgba(248, 81, 73, 0.2)' : 'transparent');
                        tableHtml += `<td style="padding: 12px 18px; font-size: 1.1rem; background: ${color}; border: 1px solid rgba(255,255,255,0.05);">${val}</td>`;
                    });
                    tableHtml += `</tr>`;
                });

                tableHtml += `</table></div>`;
                cmContainer.innerHTML = tableHtml;
            }
        }
    }
});
