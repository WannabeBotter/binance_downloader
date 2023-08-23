import argparse
import pathlib
import tarfile
import zipfile
from io import BytesIO
from typing import Tuple

import numpy as np
import polars as pl

DEPTH_EVENT = 1
TRADE_EVENT = 2
DEPTH_CLEAR_EVENT = 3
DEPTH_SNAPSHOT_EVENT = 4


def convert_trades_zip_to_polars(target_symbol: str = "BTCUSDT", target_date: str = "2023-08-22") -> pl.DataFrame:
    print("トレード履歴をzipファイルから読み込み、Polarsデータフレームに変換します")
    target_trades_zip = (
        pathlib.Path("data")
        / "trades"
        / target_symbol
        / f"{target_symbol}-trades-{target_date}.zip"
    )

    with zipfile.ZipFile(BytesIO(target_trades_zip.open("rb").read())) as _zipfile:
        if _zipfile.testzip() is not None:
            raise zipfile.BadZipFile
        _stem = pathlib.Path(target_trades_zip).stem
        _trades_csv = _zipfile.read(f"{_stem}.csv")
        _df_trades = pl.read_csv(
            BytesIO(_trades_csv),
            skip_rows=1,
            new_columns=[
                "id",
                "price",
                "qty",
                "quote_qty",
                "time",
                "is_buyer_maker"
            ],
            dtypes={
                "id": pl.UInt64,
                "price": pl.Float64,
                "qty": pl.Float64,
                "quote_qty": pl.Float64,
                "time": pl.UInt64,
                "is_buyer_maker": pl.Boolean
            },
        )
        _df_trades = _df_trades.with_columns(pl.lit(TRADE_EVENT).alias("event"))
        _df_trades = _df_trades.rename({"time": "exch_timestamp"})
        _df_trades = _df_trades.with_columns(pl.col("exch_timestamp").alias("local_timestamp"))
        _df_trades = _df_trades.with_columns(pl.when(pl.col("is_buyer_maker") == True).then(-1).otherwise(1).alias("side"))
        _df_trades = _df_trades.drop(["id", "is_buyer_maker", "quote_qty"])
        _df_trades = _df_trades.select(["event", "exch_timestamp", "local_timestamp", "side", "price", "qty"])

    return _df_trades

def convert_orderbook_targz_to_polars(target_symbol: str = "BTCUSDT", target_date: str = "2023-08-22") -> Tuple[pl.DataFrame, pl.DataFrame]:
    print("オーダーブック履歴をtar.gzファイルから読み込み、Polarsデータフレームに変換します")
    target_orderbook_targz = (
        pathlib.Path("data")
        / "orderbook"
        / target_symbol
        / f"{target_symbol}_T_DEPTH_{target_date}.tar.gz"
    )
    with tarfile.open(target_orderbook_targz, mode="r:*") as _tarfile:
        _filenames = sorted(_tarfile.getnames())
        _orderbook_snap_csv = _tarfile.extractfile(_filenames[0])
        _df_orderbook_snap = pl.read_csv(
            BytesIO(_orderbook_snap_csv.read()),
            skip_rows=1,
            new_columns=[
                "symbol",
                "timestamp",
                "trans_id",
                "first_update_id",
                "last_update_id",
                "side",
                "update_type",
                "price",
                "qty",
            ],
            dtypes={
                "symbol": pl.Utf8,
                "timestamp": pl.UInt64,
                "trans_id": pl.UInt64,
                "first_update_id": pl.UInt64,
                "last_update_id": pl.UInt64,
                "side": pl.Utf8,
                "update_type": pl.Utf8,
                "price": pl.Float64,
                "qty": pl.Float64,
            },
        )
        _df_orderbook_snap = _df_orderbook_snap.with_columns(pl.lit(DEPTH_SNAPSHOT_EVENT).alias("event"))
        _df_orderbook_snap = _df_orderbook_snap.rename({"timestamp": "exch_timestamp"})
        _df_orderbook_snap = _df_orderbook_snap.with_columns(pl.col("exch_timestamp").alias("local_timestamp"))
        _df_orderbook_snap = _df_orderbook_snap.with_columns(pl.when(pl.col("side") == "a").then(-1).otherwise(1).alias("side"))
        _df_orderbook_snap = _df_orderbook_snap.drop(["symbol", "trans_id", "first_update_id", "last_update_id", "update_type"])
        _df_orderbook_snap = _df_orderbook_snap.select(["event", "exch_timestamp", "local_timestamp", "side", "price", "qty"])

        _orderbook_update_csv = _tarfile.extractfile(_filenames[1])
        _df_orderbook_update = pl.read_csv(
            BytesIO(_orderbook_update_csv.read()),
            skip_rows=1,
            new_columns=[
                "symbol",
                "timestamp",
                "trans_id",
                "first_update_id",
                "last_update_id",
                "side",
                "update_type",
                "price",
                "qty",
            ],
            dtypes={
                "symbol": pl.Utf8,
                "timestamp": pl.UInt64,
                "trans_id": pl.UInt64,
                "first_update_id": pl.UInt64,
                "last_update_id": pl.UInt64,
                "side": pl.Utf8,
                "update_type": pl.Utf8,
                "price": pl.Float64,
                "qty": pl.Float64,
            },
        )
        _df_orderbook_update = _df_orderbook_update.with_columns(pl.lit(DEPTH_EVENT).alias("event"))
        _df_orderbook_update = _df_orderbook_update.rename({"timestamp": "exch_timestamp"})
        _df_orderbook_update = _df_orderbook_update.with_columns(pl.col("exch_timestamp").alias("local_timestamp"))
        _df_orderbook_update = _df_orderbook_update.with_columns(pl.when(pl.col("side") == "a").then(-1).otherwise(1).alias("side"))
        _df_orderbook_update = _df_orderbook_update.drop(["symbol", "trans_id", "first_update_id", "last_update_id", "update_type"])
        _df_orderbook_update = _df_orderbook_update.select(["event", "exch_timestamp", "local_timestamp", "side", "price", "qty"])

    return _df_orderbook_snap, _df_orderbook_update


def prepare_datadir(datadir: str, symbol: str, target: str) -> pathlib.Path:
    _datadir_path: pathlib.Path = pathlib.Path(f"{datadir}/{target}/{symbol}/")
    if ~_datadir_path.is_dir():
        _datadir_path.mkdir(parents=True, exist_ok=True)
    return _datadir_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', help="ダウンロードする対象の銘柄 例:BTCUSDT")
    parser.add_argument('--date', help="変換する対象の日付 例:2023-04-01")
    args = parser.parse_args()

    target_symbol = args.symbol
    target_date = args.date

    print("データをzipファイルとtar.gzファイルから読み込みます")
    df_trades = convert_trades_zip_to_polars(target_symbol, target_date)
    df_orderbook_snap, df_orderbook_update = convert_orderbook_targz_to_polars(target_symbol, target_date)
    df = df_trades.vstack(df_orderbook_snap).vstack(df_orderbook_update).sort(["exch_timestamp", "event"])
    print(df)

    np_array = df.to_numpy()
    datadir = prepare_datadir("data", target_symbol, "npz")
    print(f"データを保存します")
    np.savez(datadir / f"{target_symbol}_{target_date}.npz", arr=np_array)
    print(f"データを {datadir / f'{target_symbol}_{target_date}.npz'} に保存しました")
