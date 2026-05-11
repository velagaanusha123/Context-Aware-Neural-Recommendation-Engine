async function getRecommendations() {
    const userId = document.getElementById('userId').value;
    const resultsDiv = document.getElementById('results');

    if (!userId) {
        resultsDiv.innerHTML = '<p>Please enter a User ID</p>';
        return;
    }

    resultsDiv.innerHTML = '<p>Loading...</p>';

    try {
        const response = await fetch(`http://127.0.0.1:8000/recommend/${userId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        if (data.recommendations && data.recommendations.length > 0) {
            let html = '<h3>Recommended Movies:</h3><ul>';
            data.recommendations.forEach(movie => {

                html += `
        <li>
             ${movie.title}
             ${movie.predicted_rating.toFixed(2)}
        </li>
         `;
            });
            html += '</ul>';
            resultsDiv.innerHTML = html;
        } else {
            resultsDiv.innerHTML = '<p>No recommendations found.</p>';
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}. Is the backend running?</p>`;
        console.error('Error fetching recommendations:', error);
    }
}
