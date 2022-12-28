# saarthi-nlp-task
config folder: It contains the configuration files. It can be kept as required.
task data: It holds the data files. The train_data.xlsx is used for training. A separate split of train_data.xlxs is used for validation purpose. For testing we can put test_data.xlxs and test using the tester.py

train.py: This is the first file which is needed to be run. After setting up the environment, logging into wandb and installing required packages, simply run this file to train the model. The best model will automatically get saved.
```
$ python train.py
```
predict.ipynb: Use this notebook after training to test on custom sentences.

tester.py: After the test.csv is placed in task data, this file can be run to get the accuracy on the test data.
```
$ python tester.py
```
