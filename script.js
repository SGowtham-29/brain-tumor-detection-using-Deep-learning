document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    const fileInput = document.getElementById('imageInput').files[0];
    if (!fileInput) return alert("Please select an image.");
    
    let formData = new FormData();
    formData.append("image", fileInput);

    let response = await fetch('/predict', { method: 'POST', body: formData });
    let result = await response.json();
    
    document.getElementById('result').innerHTML = `Prediction: <strong>${result.label}</strong> (Confidence: ${result.confidence}%)`;
    
    let imgElement = document.createElement('img');
    imgElement.src = URL.createObjectURL(fileInput);
    imgElement.style.display = "block";
    
    let previewDiv = document.getElementById('imagePreviews');
    previewDiv.innerHTML = "";
    previewDiv.appendChild(imgElement);
});
