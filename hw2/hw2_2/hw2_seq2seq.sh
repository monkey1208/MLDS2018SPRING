wget https://www.csie.ntu.edu.tw/~b03902101/decoder.pt
python3 test.py --encoder model/encoder.pt --decoder decoder.pt --input $1 --output $2 
