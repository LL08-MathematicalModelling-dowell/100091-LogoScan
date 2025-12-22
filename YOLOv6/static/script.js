function callBackend() {
    fetch("/api/hello")
        .then(response => response.json())
        .then(data => {
            document.getElementById("response").innerText = data.message;
        })
        .catch(err => console.log(err));
}
