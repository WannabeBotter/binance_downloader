import argparse
import datetime
import gc
import hashlib
import hmac
import io
import os
import pathlib
import tarfile
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import polars as pl
import requests
from joblib import Parallel, delayed
from joblib_util import tqdm_joblib
from retrying import retry

S_URL_V1 = "https://api.binance.com/sapi/v1"

# API keys
api_key = "5tDfvj4KZNsz0D3mDKlB94cTLQlrbDKug8oFcuspYGOWjk4ifLVlQXqWhFWBqkVW"
secret_key = "YtXCLeLw8DZvsAAjnoh3aqfM7mYJMVESqNZK83WqBz778JhKNTvmCJF1EeT2PWq2"

# Utility functions from Binance sample
# https://github.com/binance/binance-public-data/tree/master/Futures_Order_Book_Download

# Function to generate the signature
def sign(params={}):
    data = params.copy()
    ts = str(int(1000 * time.time()))
    data.update({"timestamp": ts})
    h = urlencode(data)
    h = h.replace("%40", "@")
    b = bytearray()
    b.extend(secret_key.encode())
    signature = hmac.new(b, msg=h.encode("utf-8"), digestmod=hashlib.sha256).hexdigest()
    sig = {"signature": signature}
    return data, sig


# Function to generate the download ID
def post(path, params={}):
    sign = _sign(params)
    query = urlencode(sign[0]) + "&" + urlencode(sign[1])
    url = "%s?%s" % (path, query)
    header = {"X-MBX-APIKEY": api_key}
    resultPostFunction = requests.post(url, headers=header, timeout=30, verify=True)
    return resultPostFunction


# Function to generate the download link
def get(path, params):
    sign = _sign(params)
    query = urlencode(sign[0]) + "&" + urlencode(sign[1])
    url = "%s?%s" % (path, query)
    header = {"X-MBX-APIKEY": api_key}
    resultGetFunction = requests.get(url, headers=header, timeout=30, verify=True)
    return resultGetFunction

def prepare_datadir(datadir: str, symbol: str, target: str) -> pathlib.Path:
        _datadir_path: pathlib.Path = pathlib.Path(f"{datadir}/{target}/{symbol}/")
        if ~_datadir_path.is_dir():
            _datadir_path.mkdir(parents=True, exist_ok=True)
        return _datadir_path

# 指定されたファイル名をもとに、.zipをダウンロードしてデータフレームを作り、pkl.gzとして保存する関数
@retry(stop_max_attempt_number = 5, wait_fixed = 1000)
def download_orderbook_zip(symbol: str = None, startdate: datetime.datetime = None, enddate:datetime.datetime = None, datadir: str = None) -> None:
    assert symbol is not None
    assert startdate is not None
    assert enddate is not None
    assert datadir is not None

    _starttime = int(startdate.timestamp()) * 1000
    _endtime = int(enddate.timestamp()) * 1000
    _datatype = "T_DEPTH"

    _timestamp = str(int(1000 * time.time()))  # current timestamp which serves as an input for the params variable
    _params = {
        "symbol": symbol,
        "startTime": _starttime,
        "endTime": _endtime,
        "dataType": _datatype,
        "timestamp": _timestamp,
    }
    _sign = sign(_params)
    _url = f"{S_URL_V1}/futuresHistDataId?{urlencode(_sign[0])}&{urlencode(_sign[1])}"

    _r = requests.post(_url, headers = {"X-MBX-APIKEY": api_key}, verify = True, timeout = 30)
    if _r.status_code != requests.codes.ok:
        print(f'response.post({_url})からHTTPステータスコード {_r.status_code} が返されました。リトライします。')
        print(_r.json())
        raise Exception

    _downloadid = _r.json()["id"]

    # Calls the "get" function to obtain the download link for the specified symbol, dataType and time range combination
    while True:
        _timestamp = str(int(1000 * time.time()))  # current timestamp which serves as an input for the params variable
        _params = {"downloadId": _downloadid, "timestamp": _timestamp}
        _sign = sign(_params)
        _url = f"{S_URL_V1}/downloadLink?{urlencode(_sign[0])}&{urlencode(_sign[1])}"

        _r = requests.get(_url, headers={"X-MBX-APIKEY": api_key}, timeout=30, verify=True)

        if _r.status_code != requests.codes.ok:
            print(f'response.get({_url})からHTTPステータスコード {_r.status_code} が返されました。このファイルをスキップします。')
            print(_r.json())
            raise Exception

        if "expirationTime" not in _r.json():
            print(f'まだダウンロードリンクが生成されていません。10秒後に再試行します。')
            print(_r.json())
            time.sleep(10)
            continue
        else:
            print(f"ダウンロードリンクを取得しました。")
            print(_r.json())
            break

    _url = _r.json()["link"]
    print(f"{_url} をダウンロードします")
    _r = requests.get(_url)
    if _r.status_code != requests.codes.ok:
        print(f'response.get({_url})からHTTPステータスコード {_r.status_code} が返されました。このファイルをスキップします。')
        raise Exception

    print("ダウンロードを完了しました")

    # Handle tar.gz file
    _fileobj = BytesIO(_r.content)
    try:
        # Process downloaded tar.gz file
        with tarfile.open(fileobj=_fileobj, mode="r:gz") as _tarfile:
            _filenames = _tarfile.getnames()
            print(_filenames)

            # Process .tar.gz files in downloaded tar.gz file
            for _filename in _filenames:
                _buffer: io.BufferedReader = _tarfile.extractfile(_filename)

                _datadir = prepare_datadir("data", symbol, "orderbook")
                _targz_path = _datadir / _filename

                with open(_targz_path, "wb") as _f:
                    _f.write(_buffer.read())
                print(f"{_targz_path} を保存しました")

    except Exception as e:
        print(e)
        raise e

    print("処理完了しました")

    return

# joblibを使って並列でダウンロードジョブを実行する関数
def download_orderbook_from_binance(symbols: list = None, startdate: str = None, enddate: str = None) -> None:
    assert symbols is not None
    assert startdate is not None
    assert enddate is not None

    _datadir = 'data/binance'
    _startdate = datetime.datetime.strptime(startdate, '%Y-%m-%d')
    _enddate = datetime.datetime.strptime(enddate, '%Y-%m-%d')
    print(f'{symbols}のオーダーブック履歴ファイルを{_startdate}から{_enddate}までダウンロードします')

    with tqdm_joblib(total = len(symbols)):
        r = Parallel(n_jobs = 4, timeout = 60*60*24)([delayed(download_orderbook_zip)(_symbol, _startdate, _enddate, _datadir) for _symbol in symbols])
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', help = 'ダウンロードする対象の銘柄 例:BTCUSDT')
    parser.add_argument('--startdate', help = 'ダウンロードする最初の日付 例:2023-04-01')
    parser.add_argument('--enddate', help = 'ダウンロードする最後の日付 例:2023-04-02')
    args = parser.parse_args()

    symbol = args.symbol
    startdate = args.startdate
    enddate = args.enddate

    if symbol:
        download_orderbook_from_binance([symbol], startdate, enddate)
    else:
        download_orderbook_from_binance(["BTCUSDT"], startdate, enddate)
