document.getElementById('prediction-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    // Get form data
    const formData = new FormData(event.target);
    const data = {};

    // Loop through all form data and populate the data object
    formData.forEach((value, key) => {
        if (key === "Discount Applied") {
            // Ensure a value for Discount Applied (1 if checked, 0 if unchecked)
            data[key] = document.getElementById('discount').checked ? 1 : 0;
        } else {
            data[key] = value;
        }
    });

    if (!data.hasOwnProperty('Discount Applied')) {
        data['Discount Applied'] = 0;  // Default to 0 if not present
    }

    // Prepare the POST request payload
    const payload = {
        features: data
    };

    console.log(payload);

    try {
        // Send POST request to Flask API
        const response = await fetch('https://test8-876322899667.asia-south1.run.app/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        // Display the prediction result
        if (result.prediction) {
            document.getElementById('prediction-output').textContent = result.prediction;
        } else {
            document.getElementById('prediction-output').textContent = "Error: " + result.error;
        }
    } catch (error) {
        document.getElementById('prediction-output').textContent = "Error: " + error.message;
    }
});
