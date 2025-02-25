import bz2
import os
import _pickle as cPickle


def save_pickled(path, data):
    """
    Save a pickled object to a file

    Args:
        path (str): path to save the file
            If the file ends with .pbz2, the data will be compressed
            If the file ends with .pkl, the data will be saved as uncompressed
        data (object): data to compress
    """
    if path.endswith(".pbz2"):
        with bz2.BZ2File(path, "w") as f:
            cPickle.dump(data, f)
    elif path.endswith(".pkl"):
        with open(path, "wb") as f:
            cPickle.dump(data, f)
    else:
        file_ext = os.path.splitext(path)[1]
        raise ValueError(
            f"Unknown file extension {file_ext}. Has to end with .pbz2 or .pkl"
        )


def extract_pickled(file):
    """Load (and decompress) a pickled object from a file

    Args:
        file (str): path to the compressed file
    """
    if file.endswith(".pbz2"):
        data = bz2.BZ2File(file, "rb")
    else:
        data = open(file, "rb")
    data = cPickle.load(data)
    return data
