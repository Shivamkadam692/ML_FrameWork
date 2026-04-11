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
    
    // Chart Instance
    let lossChart = null;
    let currentModel = 'linear';

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

            // Update description
            datasetDesc.textContent = descriptions[currentModel];
            document.getElementById('howItWorksDesc').innerHTML = howItWorks[currentModel];
        });
    });

    // Train Model Logic
    trainBtn.addEventListener('click', async () => {
        // Build Payload
        let payload = {};
        if (currentModel === 'tree') {
            payload = {
                maxDepth: parseInt(document.getElementById('maxDepth').value),
                minSamples: parseInt(document.getElementById('minSamples').value)
            };
        } else {
            payload = {
                learningRate: parseFloat(document.getElementById('learningRate').value),
                iterations: parseInt(document.getElementById('iterations').value)
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

                // Plot Loss Curve
                plotLoss(data.loss_history, currentModel);
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
});
