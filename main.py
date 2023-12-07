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

# Stremlit とコード実行順序の関連を追跡し理解するためのPRINT文（ターミナルに出力される）
global_flag = False
print('This main.py code starts.', global_flag)

# 必要なモジュールをインポートする
import streamlit as st

# 真っ先に行う処理
# タイトルを表示
st.title('数字の画像分類へようこそ')
# st.sidebar.write('V0.00 R5(2023)/11/30')
st.sidebar.write('''
    V0.04 R5(2023)/12/07
    ''')
    # V0.01 R5(2023)/12/02 \n
    # V0.02 R5(2023)/12/03 \n
    # V0.03 R5(2023)/12/06)


# 必要なモジュールをインポートする
import numpy as np
from PIL import Image

# 自作モジュールにアクセスできるようにする
import my1_cvtmnist as my_cvtmnist
import my2_cnn as my_cnn
import my3_predictX as my_predict


# 使い方の説明を表示
st.write('数字一つが写った画像をアップロードしてAIが数字を分類します。そして、どこを見て分類したのかをヒートマップで表示するAPIです。')

col1, col2 = st.columns(2)

# アップロードした画像を表示する
def display_img(uploaded_file, col):
    # # バイトデータとしてファイルを読み取るとき
    # bytes_data = upload_file.getvalue()
    # st.write(bytes_data)

    # アップロードされた画像を表示
    with col:
        st.image(uploaded_file, caption='Uploaded Image.', width=200)
        st.write("この画像から数字を予測します。")


def classify_img(uploaded_file):
    st.write('Step 3')

    # アップロードした画像をMNIST画像仕様に変換する
    digit_MNIST_img = my_cvtmnist.CvToMNIST_img(uploaded_file)
    # MNIST画像変換した画像を表示する
    st.image(digit_MNIST_img, caption='MNIST仕様画像')
    # MNIST画像に描かれている数字を予測する
    prediction = my_predict.PredictNumber(digit_MNIST_img)

    st.sidebar.write(f'予測結果： {prediction}')
    st.write(f'分類結果： {prediction.numpy()}')

    # ヒートマップ画像をローカルに保存し保存した画像ファイルを表示する
    my_predict.eXplainableAI(digit_MNIST_img)
    st.write('上段はMNISTデータセットからランダムに拾った画像で特徴量を例示し、下段がアップロードした画像での特徴量です')
    st.write('青色はマイナスに影響、赤色はプラスに影響を与えている部分らしいです')
    st.image('shap_plot white.png', caption='eXplainableAI', use_column_width=True)


def main():
    global global_flag

    # 各UIを定義するコードを最初に集める。
    # ファイルを選択するUI
    with col1:
        st.write('Step 1')
        uploaded_file = st.file_uploader('画像ファイルを選択してください。', type=['png','jpg','jpeg'])

    # サンプルファイル名を選ぶボタン
    btn_s0 = st.sidebar.button('サンプル画像0', key='btn_s0')
    btn_s1 = st.sidebar.button('サンプル画像1', key='btn_s1')
    btn_s2 = st.sidebar.button('サンプル画像2', key='btn_s2')
    btn_s3 = st.sidebar.button('サンプル画像3', key='btn_s3')
    btn_s4 = st.sidebar.button('サンプル画像4', key='btn_s4')
    btn_s5 = st.sidebar.button('サンプル画像5', key='btn_s5')
    btn_s6 = st.sidebar.button('サンプル画像6', key='btn_s6')
    btn_s7 = st.sidebar.button('サンプル画像7', key='btn_s7')
    btn_s8 = st.sidebar.button('サンプル画像8', key='btn_s8')
    btn_s9 = st.sidebar.button('サンプル画像9', key='btn_s9')
    uploaded_file_sample = None

    # 分類開始するボタン
    st.write('Step 2')
    btn_1 = st.button('分類を進めるには、ここをクリックして下さい。', key='btn_1')


    if btn_s0:
        uploaded_file_sample ='./sample_img/black00.png'
    else:
        if btn_s1:
            uploaded_file_sample ='./sample_img/black01.png'
        else:
            if btn_s2:
                uploaded_file_sample ='./sample_img/black02.png'
            else:
                if btn_s3:
                    uploaded_file_sample ='./sample_img/black03.png'
                else:
                    if btn_s4:
                        uploaded_file_sample ='./sample_img/black04.png'
                    else:
                        if btn_s5:
                            uploaded_file_sample ='./sample_img/black05.png'
                        else:
                            if btn_s6:
                                uploaded_file_sample ='./sample_img/black06.png'
                            else:
                                if btn_s7:
                                    uploaded_file_sample ='./sample_img/black07.png'
                                else:
                                    if btn_s8:
                                        uploaded_file_sample ='./sample_img/black08.png'
                                    else:
                                        if btn_s9:
                                            uploaded_file_sample ='./sample_img/black09.png'
    if uploaded_file_sample is not None:
        st.session_state['key'] = uploaded_file_sample
        print('uploaded_file_sample ',uploaded_file_sample)
        display_img(uploaded_file_sample, col2)       

    # if btn_s1:
    #     uploaded_file_sample = './sample_img/bblack01.png'
    #     if 'key' not in st.session_state:
    #         st.session_state['key'] = uploaded_file_sample

    # 選択画像を表示する
    if uploaded_file is not None:
        print('uploaded_file ',uploaded_file)
        display_img(uploaded_file, col2)

    # if btn_s1:
    #     print('uploaded_file_sample ',uploaded_file_sample)
    #     display_img(uploaded_file_sample, col2)

    print('uploaded_file 2',uploaded_file)
    print('uploaded_file_sample 2',uploaded_file_sample)


    # 分類する
    if btn_1:
        if 'key' in st.session_state:
            uploaded_file_sample = st.session_state['key']

        if uploaded_file is not None:
            classify_img(uploaded_file)
        else:
            if uploaded_file_sample is not None:
                classify_img(uploaded_file_sample)   
                # del st.session_state['key']
            else:
                st.write('画像が未選択です。')  

       
        # if classify_file is not None:
        #     print('変数のType',type(classify_file))
        #     classify_img(classify_file)
        # else:
        #     print('画像ファイルが未選択です。')


print("This main.py code ends.", global_flag)


if __name__ == '__main__':
    print('Main（）runs.')
    main()

# picture = st.camera_input("Take a picture")
# if picture:
#     st.image(picture)