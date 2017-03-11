TRAIN=../hw1_test/train.csv
TEST=../hw1_test/test_X.csv
RESULT=../hw1_test/result

all:
	./hw1.sh $(TRAIN) $(TEST) $(RESULT)/make_v2.csv

v1:
	python3 LinearRegression_v1.py $(TRAIN) $(TEST) $(RESULT)/make_v1.csv

v2:
	python3 LinearRegression_v2.py $(TRAIN) $(TEST) $(RESULT)/make_v2.csv
