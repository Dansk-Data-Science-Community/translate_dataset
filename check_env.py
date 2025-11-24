import sys

print("PYTHON:", sys.executable)
try:
    import datasets
    print("DATASETS VERSION:", datasets.__version__)
except ImportError as e:
    print("IMPORT ERROR:", e)
