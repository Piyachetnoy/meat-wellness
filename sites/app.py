#Flaskの設定
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image

from model import Model

app = Flask(__name__)

# 画像の保存先ディレクトリ
UPLOAD_FOLDER = 'static/img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# アップロードされるファイルの拡張子を指定
ALLOWED_EXTENSIONS = {'png', 'jpg'}

#ファイルが許可されている形式かの確認
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#ルートurlの処理
@app.route('/')
def index():
    return render_template('index.html')

#ファイルアップロードの処理
@app.route('/img', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'result': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'result': 'error', 'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        model = Model(path=filepath)

        # 画像URLの作成
        image_url = f'/{filepath}'

        print(model.result)

        if model.result == 1:
            message = "焼肉"
            css_class = "burned"
        elif model.result == 0:
            message = "生"
            css_class = "raw"
        else:
            message = "error"
            css_class = "Not_meat"
        
        return jsonify({'result': model.result, 'image_url': image_url, 'message':message, 'css_class':css_class})
    
    
    return jsonify({'result': 'error', 'message': 'File not allowed'})

#アプリケーションの起動
if __name__ == '__main__':
    # 必要に応じて、ポート番号を変更
    app.run(debug=True)
