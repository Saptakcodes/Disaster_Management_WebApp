// Chart.js initialization for analysis pages
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all charts on the analysis page
    if (document.getElementById('floodChart')) {
        initFloodChart();
    }
    
    if (document.getElementById('fireChart')) {
        initFireChart();
    }
    
    if (document.getElementById('earthquakeChart')) {
        initEarthquakeChart();
    }
    
    if (document.getElementById('cycloneChart')) {
        initCycloneChart();
    }
    
    // Chart type toggle functionality
    const chartTypeToggles = document.querySelectorAll('.chart-type-toggle');
    chartTypeToggles.forEach(toggle => {
        toggle.addEventListener('change', function() {
            const chartId = this.dataset.chartId;
            const chartType = this.value;
            changeChartType(chartId, chartType);
        });
    });
});

// Flood frequency chart
function initFloodChart() {
    const ctx = document.getElementById('floodChart').getContext('2d');
    window.floodChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            datasets: [{
                label: 'Flood Events',
                data: [5, 8, 12, 15, 20, 25, 28, 26, 18, 12, 8, 6],
                backgroundColor: '#388E3C',
                borderColor: '#2E7D32',
                borderWidth: 1
            }]
        },
        options: getChartOptions('Monthly Flood Frequency')
    });
}

// Fire risk vs temperature chart
function initFireChart() {
    const ctx = document.getElementById('fireChart').getContext('2d');
    window.fireChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['60°F', '65°F', '70°F', '75°F', '80°F', '85°F', '90°F', '95°F', '100°F'],
            datasets: [{
                label: 'Fire Risk Index',
                data: [15, 25, 40, 60, 75, 85, 90, 95, 98],
                backgroundColor: 'rgba(255, 99, 71, 0.2)',
                borderColor: 'rgba(255, 99, 71, 1)',
                borderWidth: 2,
                tension: 0.3
            }]
        },
        options: getChartOptions('Fire Risk vs Temperature')
    });
}

// Earthquake magnitude over time chart
function initEarthquakeChart() {
    const ctx = document.getElementById('earthquakeChart').getContext('2d');
    window.earthquakeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'],
            datasets: [{
                label: 'Average Magnitude',
                data: [4.2, 4.5, 4.8, 5.1, 5.3, 5.0, 4.9, 5.2, 5.5],
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 2
            }]
        },
        options: getChartOptions('Earthquake Magnitude Over Time')
    });
}

// Cyclone frequency chart
function initCycloneChart() {
    const ctx = document.getElementById('cycloneChart').getContext('2d');
    window.cycloneChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Atlantic', 'Pacific', 'Indian', 'Southern'],
            datasets: [{
                label: 'Cyclones by Ocean',
                data: [35, 45, 25, 5],
                backgroundColor: [
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(255, 206, 86, 0.7)',
                    'rgba(75, 192, 192, 0.7)'
                ],
                borderColor: [
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: getChartOptions('Cyclone Distribution by Ocean Basin')
    });
}

// Common chart options
function getChartOptions(title) {
    return {
        responsive: true,
        plugins: {
            title: {
                display: true,
                text: title,
                font: {
                    size: 16
                }
            },
            legend: {
                position: 'bottom'
            },
            tooltip: {
                mode: 'index',
                intersect: false
            }
        },
        scales: {
            y: {
                beginAtZero: true
            }
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
        }
    };
}

// Change chart type dynamically
function changeChartType(chartId, chartType) {
    const chart = window[`${chartId}Chart`];
    if (chart) {
        chart.config.type = chartType;
        chart.update();
    }
}