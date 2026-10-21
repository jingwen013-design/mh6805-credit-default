"""
fetch_data.py — obtain the raw UCI file and write the CSV that common.py reads.   Owner: shared.

Source: Yeh, I.-C. & Lien, C.-H. (2009), "Default of credit card clients", UCI Machine Learning Repository, dataset 350,
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients (licence CC BY 4.0).

What it does
  1. uses data/raw/default_of_credit_card_clients.xls if present, otherwise downloads the UCI zip and extracts the .xls;
  2. checks the file's SHA-256 against the value recorded below (the file the whole project was built on);
  3. reads the sheet (header row 2), renames the two UCI column names that the code base does not use
     (PAY_0 -> PAY_1, "default payment next month" -> DEFAULT) and writes data/raw/default_of_credit_card_clients.csv.
Nothing else is changed: 30,000 rows x 25 columns in, 30,000 x 25 out. common.load_raw() then cleans and splits.

Run from Project/code:  python3 fetch_data.py
"""
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

import pandas as pd

from common import RAW, ROOT

XLS = os.path.join(ROOT, "data", "raw", "default_of_credit_card_clients.xls")
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
SHA256_XLS = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"   # the file this project was built on
RENAME = {"PAY_0": "PAY_1", "default payment next month": "DEFAULT"}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if not os.path.exists(XLS):
    print(f"downloading {URL} ...")
    buf = urllib.request.urlopen(URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(buf)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".xls"))
        os.makedirs(os.path.dirname(XLS), exist_ok=True)
        open(XLS, "wb").write(z.read(name))
    print(f"saved {XLS}")

digest = sha256(XLS)
print(f"SHA-256 of the .xls: {digest}")
if digest != SHA256_XLS:
    sys.exit("the .xls does not match the file this project was built on — stop and compare before continuing")

df = pd.read_excel(XLS, header=1).rename(columns=RENAME)
expected = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + [f"PAY_{i}" for i in range(1, 7)] + \
           [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)] + ["DEFAULT"]
assert list(df.columns) == expected, list(df.columns)
assert df.shape == (30000, 25), df.shape
df.to_csv(RAW, index=False)
print(f"wrote {RAW}: {df.shape[0]:,} rows x {df.shape[1]} columns; SHA-256 {sha256(RAW)}")
