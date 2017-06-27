Machine Learning Final Project
===
### Directory Structures
* `./src`: Source Code
    * `./src/sj_list.txt`: Model list used by ensembling.
    * `./src/iq_list.txt`: Model list used by ensembling.
* `./model`: Model Directory
* `./history`: Training history and ground truth graph directory.
### Usage
* Train Neural Network
  `python3.5 ./src/NeuralNetwork.py <data directory> <prediction directory>`
  * The output file would be `<prediction directory>/sj_[val_mae]_iq_[val_mae].csv`
* Predict by Ensemble
  `python3.5 ./src/Ensemble.py <data directory> <prediction directory>`
  * The output file would be `<prediction directory>/result.csv`
