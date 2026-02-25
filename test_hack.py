import sys
import numpy
import pickle

# Hack to fix unpickling of Numpy 2.0 models in Numpy 1.x
sys.modules['numpy._core'] = numpy

try:
    with open("rfc-diabetes.pkl", "rb") as f:
        model = pickle.load(f)
    print("SUCCESS: Model loaded with hack!")
except Exception as e:
    print(f"FAILURE: Could not load model even with hack. Error: {e}")
