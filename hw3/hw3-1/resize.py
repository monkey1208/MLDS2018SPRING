import os
import cv2

input_path = 'data/0'
output_path = 'resized_data/0/'
for filename in os.listdir(input_path):
    print(filename)
    img = cv2.imread(input_path+'/'+filename)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path+filename, img)