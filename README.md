**TEAM:** Pyaaz Kachori\
**AUTHORS:** Akash Gupta, Apoorva Bharadwaj, Somya Agrawal, Sayan Biswas\
**Current Version:** 0.0.1\
**Current Implementation :** Sentiment analysis and review summarisaation using T5.

[Github Repo Link](https://github.com/gakash2k01/Cisco-sentiment-analysis/tree/kachori)

## CLONING GUIDE:
`git clone git@github.com:gakash2k01/Cisco-sentiment-analysis.git`

## FOLDERING:
create folder `task_data`=> put training and test files:\
create folder `weights`=> puts weights there

## CONFIGURING:
Edit the `config.yml` file in `config` folder, add your wandb account name and project, set learning rates and training epochs.

## REQUIREMENTS INSTALLATION:
`!pip install -r requirements.txt
`

# TRAINING:
`!python3 train.py
`

## OUTPUT GENERATION
`!python3 tester.py
`
