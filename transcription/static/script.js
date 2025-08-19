document.addEventListener("DOMContentLoaded", () => {
    async function uploadAudio() {
        const fileInput = document.getElementById('audioFile');
        const file = fileInput.files[0];
        if (!file) {
            alert("Please select a file first.");
            return;
        }

        const formData = new FormData();
        formData.append("audio", file);

        fetch("http://localhost:5001/transcribe", {
            method: "POST",
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById("responseText").textContent = "Server Response: " + data.text;
        })
        .catch(error => {
            console.error("Error:", error);
        });
    }

    const button = document.getElementById("uploadBtn");

    if (button) {
        button.addEventListener("click", uploadAudio);
    }

    window.uploadAudio = uploadAudio;
});
