# キカガク　DXを推進するAI・データサイエンス人材育成コース　給付金要件　自走期間課題の提出
# 設定課題 XAI

# 画像をMINIST仕様画像に変換するなどの画像変換を行うモジュール

# アップロード画像をMNIST仕様画像に変換して戻す。
# V1.0.0  2023/11/30
# MNIST画像仕様
# 1.切り取った画像をグレースケールに変換
# 2.グレースケールにした画像を二値化して白黒にする
# 3.白黒画像を反転し、黒背景にする
# 4.ガウスブラーをかけて補完する
# 5.28x28pxの画像にサイズ変換する
# 6.numpy配列に変換する
#
# Streamlitのfile_uploaderでアップロードされたファイルは、
# 一般的にPIL（Python Imaging Library）を使用して読み込むことができます。
#

# 必要なモジュールをインポートする
from PIL import Image
import cv2
import numpy as np


def CvToMNIST_img(uploaded_file):
    # streamlit形式(PIL)をopencv(Numpy配列かスカラー)に変換
    uploaded_image = Image.open(uploaded_file)
    freehand_image = np.array(uploaded_image)

    # グレースケール画像に変換
    im = cv2.cvtColor(freehand_image, cv2.COLOR_BGR2GRAY)
    # グレースケール画像を二値化
    _, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
    # 二値化された画像を反転
    thresh = cv2.bitwise_not(thresh)
    # ガウスブラーを適用して補完
    thresh = cv2.GaussianBlur(thresh, (9, 9), 0)
    # 画像データをリサイズ
    im_t = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_CUBIC)
    # OpenCVの画像をPIL形式に変換して戻す
    return  Image.fromarray(im_t)