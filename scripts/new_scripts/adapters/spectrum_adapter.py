from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def l1_normalize(vec: np.ndarray) -> np.ndarray:
    denom = np.sum(np.abs(vec))
    if denom == 0:
        return vec.astype(np.float32)
    out = vec / denom
    out = np.nan_to_num(out, copy=False)
    return out.astype(np.float32)


def extract_spectrum_vector(xlsx_path: str | Path, sheet_name: str = '0_derivative', target_len: int | None = None, normalize: bool = True) -> np.ndarray:
    xlsx_path = Path(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, engine='openpyxl')
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    if df.empty:
        raise ValueError(f'Empty spectrum sheet: {xlsx_path} [{sheet_name}]')

    numeric = df.apply(pd.to_numeric, errors='coerce')
    values = None

    if numeric.shape[1] >= 2:
        best_col, best_count = None, -1
        for j in range(1, numeric.shape[1]):
            count = int(np.isfinite(numeric.iloc[:, j].to_numpy()).sum())
            if count > best_count:
                best_count = count
                best_col = j
        cand = numeric.iloc[:, best_col].to_numpy()
        cand = cand[np.isfinite(cand)]
        if cand.size > 0:
            values = cand.astype(np.float32)

    if values is None:
        flat = numeric.to_numpy().reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            raise ValueError(f'No numeric values found in workbook: {xlsx_path}')
        if flat.size % 2 == 0 and np.nanmax(flat[::2]) > 100 and np.nanmin(flat[::2]) >= 400:
            values = flat[1::2].astype(np.float32)
        else:
            values = flat.astype(np.float32)

    if target_len is not None and values.size != target_len:
        x_old = np.linspace(0, 1, num=values.size, dtype=np.float32)
        x_new = np.linspace(0, 1, num=target_len, dtype=np.float32)
        values = np.interp(x_new, x_old, values).astype(np.float32)

    if normalize:
        values = l1_normalize(values)
    else:
        values = values.astype(np.float32)
    return values
