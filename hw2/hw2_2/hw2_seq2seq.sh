wget https://www.csie.ntu.edu.tw/~b03902101/decoder.pt -O decoder.pt
wget https://www.dropbox.com/s/1g83pz19jgz007p/w2v200.model.bin?dl=0 -O w2v200.model.bin
wget https://www.dropbox.com/s/ix229l9p0f98mzm/w2v200.model.bin.syn1neg.npy?dl=0 -O w2v200.model.bin.syn1neg.npy
wget https://www.dropbox.com/s/33psf3t1vtnxni8/w2v200.model.bin.wv.syn0.npy?dl=0 -O w2v200.model.bin.wv.syn0.npy
python3 test.py --encoder model/encoder.pt --decoder decoder.pt --input $1 --output $2 
