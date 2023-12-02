# キカガク　DXを推進するAI・データサイエンス人材育成コース　給付金要件　自走期間課題の提出
# 設定課題 XAI

# このアプリケーションは、自分で書いた数字１文字をスマホで撮影してアップロードすることで
# 書かれた数字を判定します。
# 2023/12/2 金曜日　大安
#
# VS CODE 設定メモ：
# 1. streamlitなどのモジュール名下部に could not resolved (黄色波線)が出る。
#    コードは実行できるのでそのままでもいいが、仮想環境毎のモジュールへのパスを設定すると消せる。
#    モジュールの場所を見つける方法：
# (venv_Streamlit) ik@mini 17_自走期間課題API % python
# Python 3.10.13 (main, Nov 30 2023, 22:13:08) [Clang 15.0.0 (clang-1500.0.40.1)] on darwin
# Type "help", "copyright", "credits" or "license" for more information.
# >>> import streamlit
# >>> print(streamlit.__file__)
# /Users/ik/Downloads/virtual_env/venv_Streamlit/lib/python3.10/site-packages/streamlit/__init__.py
# >>> 
# (ctr+D でpython終了)
# /Users/ik/Downloads/virtual_env/venv_Streamlit/lib/python3.10/site-packages/ がパス
#    VS CODEの設定を開く方法：
#    メニューから Code -> 基本設定 -> 設定　を選び設定画面を表示する。
#    extra paths で検索する
#    Python › Analysis: Extra Paths 下部の「項目の追加」ボタンをクリックする
#     /Users/ik/Downloads/virtual_env/venv_Streamlit/lib/python3.10/site-packages/　を入力してOKをクリック
#    一息すると黄色波線が消えた
#
# その他設定メモ：
# 1.コードの実行方法：
#   ik@mini 17_自走期間課題API % source ../venv_Streamlit/bin/activate     
#   (venv_Streamlit) ik@mini 17_自走期間課題API % streamlit run main.py   
# 2.コードの停止方法：
#   ctr+c で止まる
#


# 必要なモジュールをインポートする
import streamlit as st

# 真っ先に行う処理
# タイトルを表示
st.title('手描き数字の画像分類')
# st.sidebar.write('V0.00 R5(2023)/11/30')
st.sidebar.write('V0.01 R5(2023)/12/02')


# 必要なモジュールをインポートする
import numpy as np
from PIL import Image

# 自作モジュールにアクセスできるようにする
import my1_cvtmnist as my_cvtmnist
import my2_cnn as my_cnn
import my3_predictX as my_predict

# 使い方の説明を表示
st.write('数字一つが写った画像をアップロードしてAIが数字を分類します。そして、どこを見て分類したのかをヒートマップで表示するAPIです。')


# アップロードした画像を表示する
def display_img(uploaded_file):
    # # バイトデータとしてファイルを読み取るとき
    # bytes_data = upload_file.getvalue()
    # st.write(bytes_data)

    # アップロードされた画像を表示
    st.image(uploaded_file, caption='Uploaded Image.', width=200)
    st.write("")
    st.write("この画像から数字を予測します。")


def classify_img(uploaded_file):
    st.write('Step 3')

    # アップロードした画像をMNIST画像仕様に変換する
    digit_MNIST_img = my_cvtmnist.CvToMNIST_img(uploaded_file)
    # MNIST画像変換した画像を表示する
    st.image(digit_MNIST_img, caption='MNIST仕様画像')
    # MNIST画像に描かれている数字を予測する
    prediction = my_predict.PredictNumber(digit_MNIST_img)

    st.sidebar.write('予測結果',prediction)
    st.write('分類結果', str(prediction.numpy()))

    # ヒートマップ画像をローカルに保存し保存した画像ファイルを表示する
    my_predict.eXplainableAI(digit_MNIST_img)
    st.write('上段はMNISTデータセットからランダムに拾った画像で特徴量を例示し、下段がアップロードした画像での特徴量です')
    st.write('青色はマイナスに影響、赤色はプラスに影響を与えている部分らしいです')
    st.image('shap_plot white.png', caption='eXplainableAI', use_column_width=True)


# ファイルをアップロードして表示する
st.write('Step 1')
uploaded_file = st.file_uploader('画像ファイルを選択してください。', type=['png','jpg','jpeg'])
if uploaded_file is not None:
    display_img(uploaded_file)

st.write('Step 2')
btn_1 = st.button('分類を進めるには、ここをクリックして下さい。')
if btn_1:
    classify_img(uploaded_file)


# 2023/12/1
#whisperインストール
# !pip install git+https://github.com/openai/whisper.git

# btn_s1 = st.sidebar.button('画像サンプル１')
# if btn_s1:
#     uploaded_file = Image.open('bblack01.png')
#     display_img(uploaded_file)

# btn_s2 = st.sidebar.button('画像サンプル２')
# if btn_s2:
#     uploaded_file = Image.open('bblack02.png')
#     display_img(uploaded_file)