import pandas as pd
import re
from clean_gadget import clean_gadget
import os


def normalization(source):
    nor_code = []
    for fun in source["code"]:
        if isinstance(fun, str):
            lines = fun.split("\n")
            code = ""
            for line in lines:
                line = line.strip()
                line = re.sub("//.*", "", line)
                code += line + " "
            code = re.sub("/\\*.*?\\*/", "", code)
            code = clean_gadget([code])
            nor_code.append(code[0])
        else:
            nor_code.append(
                ""
            )  # or any default value you prefer for non-string entries
    return nor_code


def mutrvd():
    # train = pd.read_pickle("./dataset/trvd_train.pkl")
    test = pd.read_pickle("./dataset/trvd_test.pkl")
    # val = pd.read_pickle("./dataset/trvd_val.pkl")

    # train["code"] = normalization(train)
    if not os.path.exists("./dataset/mutrvd"):
        os.makedirs("./dataset/mutrvd")
    # train.to_pickle("./dataset/mutrvd/train.pkl")

    test["code"] = normalization(test)
    test.to_pickle("./dataset/mutrvd/test.pkl")

    # val["code"] = normalization(val)
    # val.to_pickle("./dataset/mutrvd/val.pkl")


if __name__ == "__main__":
    mutrvd()
