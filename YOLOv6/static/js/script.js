document.getElementById("uploadBtn").addEventListener("click", async () => {
    const fileInput = document.getElementById("fileInput");
    const status = document.getElementById("status");

    if (!fileInput.files.length) {
        status.textContent = "Please select an image!";
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    status.textContent = "Uploading...";

    try {
        const res = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        const data = await res.json();
        status.textContent = "Scan complete: " + JSON.stringify(data);
    } catch (error) {
        status.textContent = "Upload failed!";
    }
});
