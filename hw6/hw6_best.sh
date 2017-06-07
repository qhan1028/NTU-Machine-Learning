wget https://www.dropbox.com/s/99qonlvcyfr4id7/ml2017_hw6_model.zip
unzip -u ml2017_hw6_model.zip -d model
python3.5 ensemble.py $1 $2 model_list.txt
