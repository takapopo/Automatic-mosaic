import cv2
import sys
img = cv2.imread("検出素材.jpeg")
image_file = None
args = sys.argv
print(args)
if len(args) > 1: #実行時に渡された画像ファイル名の取得
    image_file = args[1]
cv2.imshow("color",img)
cv2.waitKey(0)
print(image_file)