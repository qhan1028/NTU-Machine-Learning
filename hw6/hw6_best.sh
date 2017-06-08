wget https://www.dropbox.com/s/7sk0929m6bydlkr/ml2017_hw6_best.zip
unzip -u ml2017_hw6_best.zip -d model
python3.5 ensemble.py $1 $2 model_list.txt
