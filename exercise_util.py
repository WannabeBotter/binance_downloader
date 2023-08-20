import contextlib
import pathlib
from typing import List

import joblib
from tqdm.auto import tqdm

# それぞれのシンボルのBinanceでの上場日
target_symbols = {
    "BTCUSDT": (2019, 9, 8),
    #    'ETHUSDT': (2019, 11, 27),
    #    'XRPUSDT': (2020, 1, 6),
    #    'BNBUSDT': (2020, 2, 10),
    #    'ADAUSDT': (2020, 1, 31),
    #    'SOLUSDT': (2020, 9, 14),
    #    'DOGEUSDT': (2020, 7, 10),
    #    'MATICUSDT': (2020, 10, 22),
    #    'AVAXUSDT': (2020, 9, 23),
    #    '1000SHIBUSDT': (2021, 5, 10),
    #    'ATOMUSDT': (2020, 2, 7),
}


def list_datafiles(
    datadir: str = "data",
    datatype: str = "trades",
    symbol: str = "BTCUSDT",
    incomplete: bool = False,
) -> List[pathlib.Path]:
    """所望のデータファイルを一覧として取得する

    Args:
        datadir (str, optional): _description_. Defaults to "data".
        datatype (str, optional): _description_. Defaults to "TRADES".
        symbol (str, optional): _description_. Defaults to "BTCUSDT".
        incomplete (bool, optional): _description_. Defaults to False.

    Returns:
        List[pathlib.Path]: _description_
    """
    _result_list: List[pathlib.Path] = []

    # データディレクトリがなければ作成する
    _temp_pathobj = pathlib.Path(datadir)
    if ~_temp_pathobj.is_dir():
        _temp_pathobj.mkdir(parents=True, exist_ok=True)

    if incomplete:
        # ダウンロード途中のファイルを探している
        _target_pattern = f"TEMP_{datatype}_{symbol}_*"
        _result_list = _result_list + [_ for _ in _temp_pathobj.glob(_target_pattern)]
    else:
        _target_pattern = f"{symbol}-{datatype}-*"
        _result_list = [_ for _ in _temp_pathobj.glob(_target_pattern)]

    return sorted(_result_list)


# joblibの並列処理のプログレスバーを表示するためのユーティリティ関数
# https://blog.ysk.im/x/joblib-with-progress-bar
@contextlib.contextmanager
def tqdm_joblib(total: int = None, **kwargs):
    pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()
