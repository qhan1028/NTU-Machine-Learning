wget https://www.dropbox.com/s/6g4x3w5b3yzpqbk/ml2017_hw6.zip
unzip -u ml2017_hw6.zip -d simple_model
python3.5 mf_test.py $1 $2 ./simple_model/mf_0.839977.h5
