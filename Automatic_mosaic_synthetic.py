#透過pngの使用も可
#注意!! アルファチャンネルを含んだ４チャンネルのデータを含んだ画像ファイルではないと、行列計算の際に次元差異が発生してエラーを吐く(mask2.png)
#ファイル名、フォルダ名にスペースを入れるとエラーが発生(sys.args内)
#実行する際はファイル名に加えて画像データ名を入力
#import
import cv2
import sys #コマンドの実行時に画像ファイルを渡すために必要

image_file = None
cascade_file = './haarcascades/haarcascade_frontalface_alt.xml' #カスケードファイル(学習データ)の指定
mask_file = 'mask2.png' #合成する画像ファイルの指定

args = sys.argv

if len(args) > 1: #実行時に渡された画像ファイル名の取得
    image_file = args[1]

image = cv2.imread(image_file) #プログラム内で使える画像データの生成
mask_image = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)


cv2.imshow('image', image) #読み取った画像を表示する
cv2.waitKey(0)


image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #ノイズを減らすためにモノクロ画像へと変換する


cascade = cv2.CascadeClassifier(cascade_file) #カスケードファイルを用いてカスケード検出器を作成

front_face_list = cascade.detectMultiScale(image_gray, minSize = (30, 30)) #顔の検出処理を行う #minSizeは顔の最小サイズ(30ピクセル以上を顔と認識する)


print(front_face_list)

if len(front_face_list): #検出された顔を赤い線で囲む
    for (x,y,w,h) in front_face_list:
        print("[x,y] = %d,%d [w,h] = %d,%d" %(x, y, w, h))
        # 置き換える画像のサイズを決める
        # 長い方の幅に合わせる
        length = w
        if length < h:
            length = h
        
        # ちょっと大きめにして位置調整(そのままの大きさだと違和感あり)
        length = int(length * 1.5)
        x = x - int((length - w) / 2)
        y = y - int((length - h) / 2)

        mask_tmp = cv2.resize(mask_image, dsize=(length, length), interpolation=cv2.INTER_LINEAR) #dst = cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
        cv2.imshow('image', mask_tmp)
        cv2.waitKey(0)
        print(mask_tmp)
        ##合成処理 
        #画像が持つ透明度のデータをのデータをもとに透明な部分の処理を飛ばす
        #4ちゃんねる(RGBα⇦)目の値を255 で割って 0-1 の比率にする。描画する画像には計算した比率を掛け、背景のほうには比率の "残り" を掛けて、両者を足し合わせる。
        image[y:length + y, x:length + x] = image[y:length + y, x:length + x] * (1 - mask_tmp[:, :, 3:] / 255) \
                                            + mask_tmp[:, :, :3] * (mask_tmp[:, :, 3:] / 255)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    cv2.imwrite('./images/out_replace.jpg', image)
    cv2.imshow('image', image)
    cv2.waitKey(0)

else:
    print('not detected')