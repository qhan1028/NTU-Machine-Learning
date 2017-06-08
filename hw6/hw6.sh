wget https://www.dropbox.com/s/6g4x3w5b3yzpqbk/ml2017_hw6.zip
unzip -u ml2017_hw6.zip -d model
python3.5 mf_test.py $1 $2 ./model/mf_0.839977.h5
