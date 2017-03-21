TRAIN=../hw1_test/train.csv
TEST=../hw1_test/test_X.csv
RESULT=../hw1_test/result

all:
	./hw1.sh $(TRAIN) $(TEST) $(RESULT)/make_sgd.csv

lr:
	python3 LinearRegression.py $(TRAIN) $(TEST) $(RESULT)/make_lr.csv

sgd:
	python3 LinearRegression_SGD.py $(TRAIN) $(TEST) $(RESULT)/make_sgd.csv

reg:
	python3 LinearRegression_REG.py $(TRAIN) $(TEST) $(RESULT)/make_reg.csv
