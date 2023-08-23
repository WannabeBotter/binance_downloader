import argparse
import pathlib
import re
import time
import urllib.parse as urlparse
import zipfile
from io import BytesIO
from typing import List

import polars as pl
import requests
import xmltodict
from joblib import Parallel, delayed
from joblib_util import tqdm_joblib
from retrying import retry


class BinanceHistoricalDataManager:
    columns = {
        "trades": {
            "columns": [
                "id",
                "price",
                "qty",
                "base_qty",
                "time",
                "is_buyer_maker"
            ],
            "dtypes": {
                "id": pl.UInt64,
                "price": pl.Float64,
                "qty": pl.Float64,
                "base_qty": pl.Float64,
                "time": pl.UInt64,
                "is_buyer_maker": pl.Boolean
            }
        },
        "bookTicker": {
            "columns": [
                "update_id",
                "best_bid_price",
                "best_bid_qty",
                "best_ask_price",
                "best_ask_qty",
                "transaction_time",
                "event_time"
            ],
            "dtypes": {
                "update_id": pl.UInt64,
                "best_bid_price": pl.Float64,
                "best_bid_qty": pl.Float64,
                "best_ask_price": pl.Float64,
                "best_ask_qty": pl.Float64,
                "transaction_time": pl.UInt64,
                "event_time": pl.UInt64
            }
        },
    }

    def __init__(self, symbols: List[str]=["BTCUSDT", "ETHUSDT"], targets: List[str] = ["trades"], datadir: str = "data"):
        # ダウンロード対象とするシンボルリスト
        self.symbols: List[str] = symbols

        # ダウンロード対象とするヒストリカルデータの種類のリスト
        # aggTrades, bookDepth, bookTicker, indexPriceKlines, klines, liquidationSnapshot,
        # markpriceKlines, metrics, premiumIndexKlines, tradesのいずれか
        self.targets: List[str] = targets

        # データディレクトリ名
        self.datadir: str = datadir

    @retry(stop_max_attempt_number=5, wait_fixed=1000)
    def get_all_zipfiles(self, symbol: str, target: str) -> List[str]:
        """対象のシンボルとヒストリカルデータの種類でBinanceからダウンロード可能な全zipファイルの一覧を取得する

        Args:
            symbol (str): 対象となるシンボル
            target (str): 対象となるヒストリカルデータの種類

        Returns:
            List[str]: zipファイルのリスト
        """
        _delimiter = "/"
        _prefix = urlparse.quote(f"data/futures/um/daily/{target}/{symbol}/")
        _marker = ""

        _list_zipfiles = []

        while True:
            _url = f"https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter={_delimiter}&prefix={_prefix}"
            if _marker != "":
                _url += f"&marker={_marker}"
            response = requests.get(_url)
            if response.status_code != 200:
                raise Exception("Failed to get zip files.")

            _content_dict = xmltodict.parse(response.content.decode("utf-8"))
            _list_contents = _content_dict["ListBucketResult"]["Contents"]
            _istruncated = _content_dict["ListBucketResult"]["IsTruncated"]

            for _content in _list_contents:
                _filename = pathlib.PurePath(_content["Key"]).name
                if _filename.endswith(".zip"):
                    _list_zipfiles.append(_filename)
            if _istruncated == "false":
                break
            else:
                _marker = urlparse.quote(_content_dict["ListBucketResult"]["NextMarker"])
        return sorted(_list_zipfiles)

    def get_downloaded_zipfiles(self, symbol: str, target: str) -> List[str]:
        """ダウンロード済みのzipファイル一覧を取得する

        Args:
            symbol (str): 対象となるシンボル
            target (str): 対象となるヒストリカルデータの種類

        Returns:
            List[str]: ダウンロード済みのzipファイルのリスト
        """
        _datadir_path = self.prepare_datadir(symbol, target)

        _target_pattern = f"{symbol}-{target}-*"
        _result_list = [_ for _ in _datadir_path.glob(_target_pattern)]

        _list_downloaded_zipfiles = []
        for _f in _result_list:
            _filename = _f.name
            _list_downloaded_zipfiles.append(_filename.replace(".parquet", ".zip"))

        return sorted(_list_downloaded_zipfiles)

    def get_downloadable_zipfiles(self, symbol: str, target: str) -> List[str]:
        """対象のシンボルとヒストリカルデータの種類で未ダウンロードのzipファイル名の一覧を取得する
        Args:
            symbol (str): 対象となるシンボル
            target (str): 対象となるヒストリカルデータの種類
        Returns:
            List[str]: 未ダウンロードのzipファイルのリスト
        """
        _list_all_zipfiles = self.get_all_zipfiles(symbol, target)
        _list_downloaded_zipfiles = self.get_downloaded_zipfiles(symbol, target)
        print(f"全ファイル数 : {len(_list_all_zipfiles)}")
        print(f"ダウンロード済みファイル数 : {len(_list_downloaded_zipfiles)}")
        return sorted(list(set(_list_all_zipfiles) - set(_list_downloaded_zipfiles)))

    @retry(stop_max_attempt_number=5, wait_fixed=1000)
    def download_zipfile(self, target_zipfile: str = None) -> None:
        assert target_zipfile is not None, "target_zipfileを指定してください。"
        """ 指定されたzipファイル名をもとに、.zipをダウンロードしてデータフレームを作り、parquetとして保存する関数。
        ダウンロードリトライは5回まで行う。

        Raises:
            Exception: _description_
            e: _description_
        """

        _m = re.match(r"(\w+)-(\w+)-*", target_zipfile)
        _symbol = _m.group(1)
        _target = _m.group(2)
        _stem = pathlib.Path(target_zipfile).stem
        _datadir_path = self.prepare_datadir(_symbol, _target)

        _url = f"https://data.binance.vision/data/futures/um/daily/{_target}/{_symbol}/{target_zipfile}"

        # ファイルをインターネットからメモリ上にダウンロード
        _r = requests.get(_url)
        if _r.status_code != requests.codes.ok:
            print(
                f"response.get({_url})からHTTPステータスコード {_r.status_code} が返されました。このファイルをスキップします。"
            )
            time.sleep(1)
            return

        _tempfile_path = _datadir_path / f"TEMP_{target_zipfile}"
        _datafile_path = _datadir_path / target_zipfile
        with open(_tempfile_path, "wb") as _f:
            _f.write(_r.content)
            _tempfile_path.rename(_datafile_path)
        return

    def download_historicaldata(self) -> None:
        """joblibを使って4並列でダウンロードジョブを実行する関数

        Args:
            datadir (str, optional): データファイルを保存するディレクトリ名。デフォルトは"data"。
            symbol (str, optional): ダウンロードする対象の銘柄ティッカー。デフォルトは"BTCUSDT".
        """
        for _symbol in self.symbols:
            for _target in self.targets:
                # データディレクトリを準備する
                _datadir_path = self.prepare_datadir(_symbol, _target)

                # 処理開始前に全ての未完了ファイルを削除する
                self.remove_incomplete_files(_datadir_path)

                _list_downloadable_zipfiles = self.get_downloadable_zipfiles(_symbol, _target)
                _num_files = len(_list_downloadable_zipfiles)
                print(f"{_symbol}の{_target}ファイルを{_num_files}個ダウンロードします")

                with tqdm_joblib(total=_num_files):
                    r = Parallel(n_jobs=4, timeout=60 * 60 * 24)(
                        [
                            delayed(self.download_zipfile)(_f)
                            for _f in _list_downloadable_zipfiles
                        ]
                    )

                # 処理開始後にも全ての未完了ファイルを削除する
                self.remove_incomplete_files(_datadir_path)

    def prepare_datadir(self, symbol: str, target: str) -> pathlib.Path:
        _datadir_path: pathlib.Path = pathlib.Path(f"{self.datadir}/{target}/{symbol}/")
        if ~_datadir_path.is_dir():
            _datadir_path.mkdir(parents=True, exist_ok=True)
        return _datadir_path

    def remove_incomplete_files(self, datadir_path: pathlib.Path) -> None:
        """未完了ファイルを削除する関数

        Args:
            datadir_path (pathlib.Path): 削除対象のディレクトリパス
        """
        _target_pattern = f"TEMP_*"
        for _f in datadir_path.glob(_target_pattern):
            _f.unlink()


# 引数処理とダウンロード関数の起動部分
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="ダウンロードする対象の銘柄 例:BTCUSDT")
    args = parser.parse_args()

    _binance_historicaldatamanager = BinanceHistoricalDataManager(symbols=["BTCUSDT"], targets=["trades"], datadir="data")
    _binance_historicaldatamanager.download_historicaldata()
