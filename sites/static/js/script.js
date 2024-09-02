document.getElementById('uploadButton').addEventListener('click', function() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('preview').src = e.target.result;
        };
        reader.readAsDataURL(file);

        const formData = new FormData();
        formData.append('file', file);

        fetch('/img', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.result !== 'error') {
                document.getElementById('result').innerText = data.message;
                console.log(data.css_class);  // これでクラスが正しく取得されているかを確認
                resultElement.className = data.css_class;


                // 画像プレビューを表示
                const imgElement = document.getElementById('preview');
                imgElement.src = data.image_url;
                imgElement.style.display = 'block';
            } else {
                document.getElementById('result').innerText = data.message;
            }
        })
        .catch(error => console.error('Error:', error));
    }
});
