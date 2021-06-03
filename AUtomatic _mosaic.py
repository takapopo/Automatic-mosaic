#import
import cv2
import sys #コマンドの実行時に画像ファイルを渡すために必要


image_file = None
cascade_file = './haarcascades/haarcascade_frontalface_alt.xml' #カスケードファイル(学習データ)の指定


args = sys.argv

if len(args) > 1: #実行時に渡された画像ファイル名の取得
    image_file = args[1]

image = cv2.imread(image_file) #プログラム内で使える画像データの生成

cv2.imshow('image', image) #読み取った画像を表示する
cv2.waitKey(0)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #ノイズを減らすためにモノクロ画像へと変換する

cv2.imshow('image', image_gray) #モノクロ化した画像を表示
cv2.waitKey(0)

cascade = cv2.CascadeClassifier(cascade_file) #カスケードファイルを用いてカスケード検出器を作成

front_face_list = cascade.detectMultiScale(image_gray, minSize = (30, 30)) #顔の検出処理を行う #minSizeは顔の最小サイズ(30ピクセル以上を顔と認識する)


print(front_face_list)

if len(front_face_list): #検出された顔を赤い線で囲む
    for (x,y,w,h) in front_face_list:
        print("[x,y] = %d,%d [w,h] = %d,%d" %(x, y, w, h))
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), thickness=10)
    
    cv2.imshow('image', image)
    cv2.waitKey(0)
else:
    print('not detected')
