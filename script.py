
!pip install -r requirements.txt

import wandb
wandb.login(key=[API KEY])

!python3 train.py

!python3 tester.py