import os
import calendar
from datetime import datetime, timedelta

import cdsapi
import xarray as xr
import numpy as np
import pandas as pd

# === 配置 ===
OUT_DIR = "./out_daily_nc"            # 每日 NC 输出目录
TMP_DIR = "./tmp_monthly_nc"          # 月度临时文件
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# 时间范围
START = datetime(2021, 7, 1)
END   = datetime(2022, 7, 31)

# 区域与分辨率（North, West, South, East）
AREA = [25.7, 109.5, 20.1, 117.4]
GRID = [0.25, 0.25]

# ERA5 单层变量（无 RH；改为下载 d2m 用于计算 RH）
ERA5_VARS = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "100m_u_component_of_wind", "100m_v_component_of_wind",
    "2m_temperature", "2m_dewpoint_temperature",
    "surface_pressure", "total_precipitation",
    "total_cloud_cover", "medium_cloud_cover",
    "low_cloud_cover", "high_cloud_cover",
]

def calc_rh_from_t_tdew(t_c, td_c):
    """
    由温度(°C)和露点(°C)计算相对湿度(%)；使用常见Magnus公式。
    RH = 100 * e(Td)/es(T)
    """
    # 饱和水汽压（hPa）
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    e  = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))
    rh = 100.0 * (e / es)
    return np.clip(rh, 0.0, 100.0)

def month_iter(start_dt, end_dt):
    y, m = start_dt.year, start_dt.month
    while (y < end_dt.year) or (y == end_dt.year and m <= end_dt.month):
        yield y, m
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1

def retrieve_month(c, year, month, target_path):
    ndays = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, ndays + 1)]
    hours = [f"{h:02d}:00" for h in range(24)]

    req = {
        "product_type": "reanalysis",
        "variable": ERA5_VARS,
        "year": f"{year:04d}",
        "month": f"{month:02d}",
        "day": days,
        "time": hours,
        "area": AREA,
        "grid": GRID,
        "format": "netcdf",
    }
    c.retrieve("reanalysis-era5-single-levels", req, target_path)

def process_month_to_daily(month_path):
    # 读取月度 NetCDF
    ds = xr.open_dataset(month_path)

    # 变量名（ERA5 NetCDF 常见命名）
    # u10/v10/u100/v100/t2m/d2m/sp/tp/tcc/mcc/lcc/hcc
    # 单位标准化：t2m->°C、tp->mm，新增 rh(%)
    # 注意：ERA5 的云量是 0-1 的比例，无需换算
    if "t2m" in ds:
        t2m_c = ds["t2m"] - 273.15
    else:
        raise KeyError("t2m not found in dataset")

    if "d2m" in ds:
        d2m_c = ds["d2m"] - 273.15
    else:
        raise KeyError("d2m (2m dewpoint) not found — needed to compute RH")

    rh = xr.apply_ufunc(
        calc_rh_from_t_tdew,
        t2m_c,
        d2m_c,
        dask="parallelized",
        output_dtypes=[float],
    ).astype("float32")
    rh.name = "rh"
    rh.attrs.update({"long_name": "2m relative humidity", "units": "%"})

    # 温度单位改为 °C
    t2m = t2m_c.astype("float32")
    t2m.name = "t2m"
    t2m.attrs.update({"long_name": "2m temperature", "units": "degC"})

    # 降水改为 mm
    tp_mm = (ds["tp"] * 1000.0).astype("float32")
    tp_mm.name = "tp"
    tp_mm.attrs.update({"long_name": "total precipitation", "units": "mm"})

    # 组装输出数据集（保留常用变量名）
    out_vars = {
        "u10": ds["u10"].astype("float32"),
        "v10": ds["v10"].astype("float32"),
        "u100": ds["u100"].astype("float32"),
        "v100": ds["v100"].astype("float32"),
        "t2m": t2m,
        "sp": ds["sp"].astype("float32"),
        "tp": tp_mm,
        "tcc": ds["tcc"].astype("float32"),
        "mcc": ds["mcc"].astype("float32"),
        "lcc": ds["lcc"].astype("float32"),
        "hcc": ds["hcc"].astype("float32"),
        "rh": rh,
    }
    ds_out = xr.Dataset(out_vars, coords=ds.coords)

    # 每天切分并保存
    time_index = pd.to_datetime(ds_out["time"].values)
    unique_days = pd.to_datetime(time_index.date).unique()

    # 压缩编码（可选）
    enc = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    # time 用默认编码即可

    for day in unique_days:
        day = pd.Timestamp(day)
        t0 = day
        t1 = day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        dsd = ds_out.sel(time=slice(t0, t1))

        # 只保存当日数据（有时可能跨月边界）
        if dsd.sizes.get("time", 0) == 0:
            continue

        out_name = f"era5_{day.strftime('%Y%m%d')}.nc"
        out_path = os.path.join(OUT_DIR, out_name)

        dsd.to_netcdf(out_path, encoding=enc)

def main():
    c = cdsapi.Client()  # 读取 ~/.cdsapirc 的 url/key

    for y, m in month_iter(START, END):
        tmp_path = os.path.join(TMP_DIR, f"era5_{y}{m:02d}.nc")
        if not os.path.exists(tmp_path):
            print(f"[CDS] downloading {y}-{m:02d} ...")
            retrieve_month(c, y, m, tmp_path)
        else:
            print(f"[CDS] already exists: {tmp_path}")

        print(f"[PROC] splitting to daily NC for {y}-{m:02d} ...")
        process_month_to_daily(tmp_path)

    print(f"Done. Daily files at: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()