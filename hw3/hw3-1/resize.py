import os
import cv2

input_path = '/2t/ylc/MLDS/hw3/data/extra_data/images'
output_path = '/2t/ylc/MLDS/hw3/data/resized_data/0/'
for filename in os.listdir(input_path):
    print(filename)
    img = cv2.imread(input_path+'/'+filename)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path+'/extra_'+filename, img)
