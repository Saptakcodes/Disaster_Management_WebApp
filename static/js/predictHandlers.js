// Generic prediction handler for all disaster types
document.addEventListener('DOMContentLoaded', function() {
    // Find all prediction forms on the page
    const predictionForms = document.querySelectorAll('.prediction-form');
    
    predictionForms.forEach(form => {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(form);
            const disasterType = form.dataset.disasterType;
            const resultContainer = document.getElementById('prediction-result');
            const loadingIndicator = document.getElementById('loading-indicator');
            
            // Show loading indicator
            resultContainer.innerHTML = '';
            loadingIndicator.classList.remove('hidden');
            
            try {
                // Simulate API call with timeout
                await new Promise(resolve => setTimeout(resolve, 1500));
                
                // Generate mock response based on disaster type
                const mockResponse = generateMockResponse(disasterType, formData);
                
                // Display result
                displayPredictionResult(mockResponse, disasterType, resultContainer);
            } catch (error) {
                resultContainer.innerHTML = `
                    <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4">
                        <p>Error: ${error.message}</p>
                    </div>
                `;
            } finally {
                loadingIndicator.classList.add('hidden');
            }
        });
    });
    
    // Generate mock API response
    function generateMockResponse(disasterType, formData) {
        const riskLevels = ['Low', 'Medium', 'High'];
        const randomRisk = riskLevels[Math.floor(Math.random() * riskLevels.length)];
        
        // Base response structure
        const response = {
            success: true,
            prediction: {
                risk: randomRisk,
                confidence: Math.floor(Math.random() * 30) + 70, // 70-100%
                timestamp: new Date().toISOString()
            },
            recommendations: []
        };
        
        // Add disaster-specific details
        switch(disasterType) {
            case 'flood':
                response.prediction.waterLevel = (Math.random() * 10).toFixed(1) + 'm';
                response.recommendations = [
                    "Monitor local water levels regularly.",
                    "Prepare sandbags if in flood-prone area.",
                    "Have an evacuation plan ready."
                ];
                break;
                
            case 'fire':
                response.prediction.temperature = (Math.random() * 20 + 70).toFixed(1) + '°F';
                response.prediction.humidity = Math.floor(Math.random() * 40) + '%';
                response.recommendations = [
                    "Clear dry vegetation around property.",
                    "Have emergency supplies ready.",
                    "Stay informed about fire department alerts."
                ];
                break;
                
            case 'earthquake':
                response.prediction.magnitude = (Math.random() * 4 + 2).toFixed(1);
                response.recommendations = [
                    "Secure heavy furniture and appliances.",
                    "Identify safe spots in each room.",
                    "Prepare an emergency kit with supplies."
                ];
                break;
                
            // Add cases for other disaster types...
                
            default:
                response.recommendations = [
                    "Stay informed through official channels.",
                    "Prepare an emergency kit with essentials.",
                    "Have an evacuation plan ready."
                ];
        }
        
        return response;
    }
    
    // Display prediction result
    function displayPredictionResult(response, disasterType, container) {
        if (!response.success) {
            container.innerHTML = `
                <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4">
                    <p>Prediction failed. Please try again later.</p>
                </div>
            `;
            return;
        }
        
        const { risk, confidence } = response.prediction;
        let riskColor, icon;
        
        // Set color and icon based on risk level
        switch(risk) {
            case 'High':
                riskColor = 'bg-red-100 text-red-800 border-red-500';
                icon = 'fa-exclamation-triangle animate-pulse';
                break;
            case 'Medium':
                riskColor = 'bg-yellow-100 text-yellow-800 border-yellow-500';
                icon = 'fa-exclamation-circle';
                break;
            default:
                riskColor = 'bg-green-100 text-green-800 border-green-500';
                icon = 'fa-check-circle';
        }
        
        // Create result card
        let resultHTML = `
            <div class="border-l-4 ${riskColor} p-6 rounded-lg shadow mb-6">
                <div class="flex items-start">
                    <i class="fas ${icon} text-2xl mr-4 mt-1"></i>
                    <div>
                        <h3 class="text-xl font-bold mb-2">${disasterType.charAt(0).toUpperCase() + disasterType.slice(1)} Risk: ${risk}</h3>
                        <p class="mb-2">Confidence: ${confidence}%</p>
        `;
        
        // Add disaster-specific details
        for (const [key, value] of Object.entries(response.prediction)) {
            if (!['risk', 'confidence', 'timestamp'].includes(key)) {
                resultHTML += `<p class="mb-1">${key.replace(/([A-Z])/g, ' $1').trim()}: ${value}</p>`;
            }
        }
        
        resultHTML += `
                    </div>
                </div>
            </div>
            
            <div class="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-lg shadow">
                <h4 class="font-bold mb-3 text-blue-800">Safety Recommendations</h4>
                <ul class="list-disc pl-5 space-y-2">
        `;
        
        // Add recommendations
        response.recommendations.forEach(rec => {
            resultHTML += `<li>${rec}</li>`;
        });
        
        resultHTML += `
                </ul>
            </div>
        `;
        
        container.innerHTML = resultHTML;
    }
});