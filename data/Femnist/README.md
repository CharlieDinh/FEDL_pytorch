# EMNIST Dataset

## Setup Instructions
- pip3 install numpy
- pip3 install pillow
- Run ```./preprocess.sh``` with a choice of the following tags:
    - ```-s``` := 'iid' to sample in an i.i.d. manner, or 'niid' to sample in a non-i.i.d. manner; more information on i.i.d. versus non-i.i.d. is included in the 'Notes' section
    - ```--iu``` := number of users, if iid sampling; expressed as a fraction of the total number of users; default is 0.01
    - ```--sf``` := fraction of data to sample, written as a decimal; default is 0.1
    - ```-k``` := minimum number of samples per user
    - ```-t``` := 'user' to partition users into train-test groups, or 'sample' to partition each user's samples into train-test groups
    - ```--tf``` := fraction of data in training set, written as a decimal; default is 0.9
    - ```--nu``` := The total number of users generated.

Instruction used to generate EMNIST with 50 users:

```
./preprocess.sh -s niid --sf 1.0 -k 0 -tf 0.8 -t sample --nu 100
```




(Make sure to delete the rem\_user\_data, sampled\_data, test, and train subfolders in the data directory before re-running preprocess.sh.)

Or you can download the dataset [here](https://drive.google.com/open?id=1sHzD4IsgEI5xLy6cqwUjSGW0PwiduPHr), unzip it and put the `train` and `test` folder under `data`.
