# settings.py
from dotenv import load_dotenv
load_dotenv(verbose=True)

import os
TRAINING_KEY = os.getenv("TRAINING_KEY")
PREDICTION_KEY= os.getenv("PREDICTION_KEY")