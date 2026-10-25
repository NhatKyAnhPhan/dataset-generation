# -*- coding: utf-8 -*-
"""
Sinh 500 triệu dòng dữ liệu ecommerce trên Windows.
Đầu vào mặc định: online_retail_II.xlsx trong cùng thư mục.
Đầu ra: online_retail_500m/folder_1 ... folder_5, mỗi folder 100 triệu dòng.
"""

import argparse
import gc
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def find_online_retail_columns(source):
    """Trả về tên cột Invoice và InvoiceDate nếu nguồn là Online Retail II."""
    columns = {
        " ".join(str(column).strip().casefold().split()): column
        for column in source.columns
    }
    required = ("invoice", "invoicedate", "quantity", "price", "stockcode")
    if all(column in columns for column in required):
        return (
            columns["invoice"],
            columns["invoicedate"],
            columns["stockcode"],
        )
    return None


def log(message):
    if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}", flush=True)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def preprocess_online_retail(source, columns):
    """Clean Online Retail II and return a compact, consistent transaction schema."""
    invoice_column, date_column, product_column = columns
    normalized_columns = {
        " ".join(str(column).strip().casefold().split()): column
        for column in source.columns
    }
    quantity_column = normalized_columns["quantity"]
    price_column = normalized_columns["price"]

    def optional_column(name, default):
        column = normalized_columns.get(name)
        if column is None:
            return pd.Series(default, index=source.index)
        return source[column]

    data = pd.DataFrame(index=source.index)
    data["invoice_id"] = source[invoice_column].astype("string").str.strip()
    data["product_id"] = source[product_column].astype("string").str.strip()
    data["description"] = (
        optional_column("description", "Unknown product")
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("Unknown product")
    )
    data["quantity"] = pd.to_numeric(
        source[quantity_column], errors="coerce"
    )
    data["event_time"] = pd.to_datetime(
        source[date_column], errors="coerce"
    )
    data["unit_price"] = pd.to_numeric(
        source[price_column], errors="coerce"
    )
    customer_column = normalized_columns.get("customer id")
    if customer_column is None:
        data["customer_id"] = pd.Series(pd.NA, index=source.index, dtype="Int64")
    else:
        data["customer_id"] = pd.to_numeric(
            source[customer_column], errors="coerce"
        ).round().astype("Int64")
    data["country"] = (
        optional_column("country", "Unknown")
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("Unknown")
    )

    valid_rows = (
        data["invoice_id"].notna()
        & data["invoice_id"].ne("")
        & data["product_id"].notna()
        & data["product_id"].ne("")
        & data["event_time"].notna()
        & data["quantity"].notna()
        & data["unit_price"].notna()
        & data["unit_price"].ge(0)
        & data["quantity"].ne(0)
    )
    removed_rows = int((~valid_rows).sum())
    data = data.loc[valid_rows].copy()
    data["quantity"] = data["quantity"].round().astype("int64")
    data["unit_price"] = data["unit_price"].astype("float64")
    data.reset_index(drop=True, inplace=True)
    return data, removed_rows


def load_source(input_file, sheet_name=None):
    """Đọc CSV hoặc Excel; mặc định ghép tất cả sheet trong workbook."""
    suffix = input_file.suffix.casefold()
    if suffix == ".csv":
        if sheet_name:
            raise ValueError("--sheet chỉ dùng được khi đầu vào là file Excel.")
        return pd.read_csv(input_file, low_memory=False)

    if suffix not in (".xlsx", ".xlsm"):
        raise ValueError("Định dạng đầu vào phải là .csv, .xlsx hoặc .xlsm.")

    try:
        with pd.ExcelFile(input_file) as workbook:
            sheets = [sheet_name] if sheet_name else workbook.sheet_names
            missing_sheets = [name for name in sheets if name not in workbook.sheet_names]
            if missing_sheets:
                raise ValueError(
                    f"Không tìm thấy sheet {missing_sheets}. "
                    f"Các sheet có sẵn: {workbook.sheet_names}"
                )

            frames = [
                pd.read_excel(workbook, sheet_name=name)
                for name in sheets
            ]
    except ImportError as error:
        raise ImportError(
            "Đọc file Excel cần openpyxl. Hãy cài bằng: pip install openpyxl"
        ) from error

    expected_columns = frames[0].columns
    if any(not frame.columns.equals(expected_columns) for frame in frames[1:]):
        raise ValueError("Các sheet được ghép phải có cùng cấu trúc cột.")
    return pd.concat(frames, ignore_index=True)


def generate_online_retail_chunk(
    source,
    n_rows,
    start_record_id,
    valid_dates,
):
    """Sample cleaned transactions and synthesize unique IDs and event times."""
    rng = np.random.default_rng()
    data = source.sample(
        n=n_rows,
        replace=True,
        random_state=int(rng.integers(0, 1_000_000_000))
    ).reset_index(drop=True).copy()

    invoice_codes, invoice_values = pd.factorize(
        data["invoice_id"],
        sort=False
    )
    invoice_codes = invoice_codes.astype("int64")
    missing_invoice = invoice_codes < 0
    group_count = len(invoice_values)
    if missing_invoice.any():
        invoice_codes[missing_invoice] = np.arange(
            group_count,
            group_count + missing_invoice.sum()
        )
        group_count += int(missing_invoice.sum())

    source_invoice_values = data["invoice_id"].astype("string")
    is_cancelled = source_invoice_values.str.startswith(
        "C",
        na=False
    ).to_numpy()
    generated_invoice_ids = (
        start_record_id + invoice_codes
    ).astype(str)
    data["invoice_id"] = np.where(
        is_cancelled,
        "C_SYN_" + generated_invoice_ids,
        "SYN_" + generated_invoice_ids
    )

    sampled_dates = rng.choice(valid_dates, size=group_count, replace=True)
    random_seconds = rng.integers(0, 24 * 60 * 60, size=group_count)
    generated_dates = (
        pd.to_datetime(sampled_dates)
        + pd.to_timedelta(random_seconds, unit="s")
    )
    data["event_time"] = generated_dates.take(invoice_codes).to_numpy()
    data.insert(
        0,
        "record_id",
        np.arange(start_record_id, start_record_id + n_rows, dtype=np.int64),
    )
    data["line_total"] = data["quantity"] * data["unit_price"]
    data["is_return"] = data["quantity"] < 0

    return data


def generate_chunk(source, n_rows, start_order_id):
    """
    Sinh n_rows dữ liệu tổng hợp từ source.
    start_order_id: số bắt đầu cho order_id để đảm bảo không trùng.
    """

    rng = np.random.default_rng()

    # ============================================================
    # 1. Lấy mẫu từ dữ liệu gốc
    # ============================================================
    data = source.sample(
        n=n_rows,
        replace=True,
        random_state=int(rng.integers(0, 1_000_000_000))
    ).reset_index(drop=True).copy()

    # ============================================================
    # 2. Tạo order_id mới - INTEGER SEQUENCE AN TOÀN
    # ============================================================
    data["order_id"] = [
        f"ORD_{start_order_id + i:010d}"
        for i in range(n_rows)
    ]

    # ============================================================
    # 3. Xử lý order_date
    # ============================================================
    source_order_dates = pd.to_datetime(
        source["order_date"],
        errors="coerce"
    ).dropna()

    if len(source_order_dates) == 0:
        raise ValueError(
            "Không tìm thấy order_date hợp lệ trong dữ liệu gốc."
        )

    sampled_dates = rng.choice(
        source_order_dates.values,
        size=n_rows,
        replace=True
    )

    data["order_date"] = pd.to_datetime(sampled_dates)

    # Thêm một khoảng thời gian ngẫu nhiên để tránh
    # tất cả record bị trùng timestamp
    random_seconds = rng.integers(
        0,
        24 * 60 * 60,
        size=n_rows
    )

    data["order_date"] = (
        data["order_date"]
        + pd.to_timedelta(random_seconds, unit="s")
    )

    # ============================================================
    # 4. Tạo các trường thời gian từ order_date
    # ============================================================
    data["order_year"] = data["order_date"].dt.year.astype("int64")
    data["order_month"] = data["order_date"].dt.month.astype("int64")
    data["order_day"] = data["order_date"].dt.day.astype("int64")
    data["order_hour"] = data["order_date"].dt.hour.astype("int64")
    data["order_minute"] = data["order_date"].dt.minute.astype("int64")
    data["order_second"] = data["order_date"].dt.second.astype("int64")

    data["is_weekend"] = np.where(
        data["order_date"].dt.dayofweek >= 5,
        "Yes",
        "No"
    )

    # ============================================================
    # 5. account_creation_date
    # ============================================================
    source_account_dates = pd.to_datetime(
        source["account_creation_date"],
        errors="coerce"
    )

    source_order_dates_full = pd.to_datetime(
        source["order_date"],
        errors="coerce"
    )

    account_age_days = (
        source_order_dates_full - source_account_dates
    ).dt.days

    account_age_days = account_age_days[
        account_age_days.notna() &
        (account_age_days >= 0)
    ]

    if len(account_age_days) > 0:
        sampled_age = rng.choice(
            account_age_days.values,
            size=n_rows,
            replace=True
        )

        data["account_creation_date"] = (
            data["order_date"]
            - pd.to_timedelta(sampled_age, unit="D")
        )
    else:
        data["account_creation_date"] = pd.NaT

    # ============================================================
    # 6. Các trường số nguyên
    # ============================================================

    integer_columns = [
        "order_year",
        "order_month",
        "order_day",
        "order_hour",
        "order_minute",
        "order_second",
        "age",
        "total_orders_by_customer",
        "product_reviews_count",
        "stock_quantity",
        "quantity",
        "discount_percent",
        "delivery_days",
        "rating",
        "pages_visited"
    ]

    for col in integer_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            ).fillna(0).round().astype("int64")

    # ============================================================
    # 7. Các trường float
    # ============================================================

    float_columns = [
        "customer_loyalty_score",
        "product_rating_avg",
        "unit_price_usd",
        "discount_amount_usd",
        "total_price_usd",
        "cost_usd",
        "profit_usd",
        "tax_usd",
        "shipping_cost_usd",
        "session_duration_minutes",
        "fraud_risk_score",
        "profit_margin_percent"
    ]

    for col in float_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            ).astype("float64")

    # ============================================================
    # 8. Đảm bảo các giá trị số hợp lệ
    # ============================================================

    if "age" in data:
        data["age"] = data["age"].clip(1, 100).astype("int64")

    if "rating" in data:
        data["rating"] = data["rating"].clip(1, 5).astype("int64")

    if "quantity" in data:
        data["quantity"] = data["quantity"].clip(1, 100).astype("int64")

    if "discount_percent" in data:
        data["discount_percent"] = (
            data["discount_percent"]
            .clip(0, 100)
            .astype("int64")
        )

    if "delivery_days" in data:
        data["delivery_days"] = (
            data["delivery_days"]
            .clip(0, 60)
            .astype("int64")
        )

    if "pages_visited" in data:
        data["pages_visited"] = (
            data["pages_visited"]
            .clip(1, 1000)
            .astype("int64")
        )

    if "product_reviews_count" in data:
        data["product_reviews_count"] = (
            data["product_reviews_count"]
            .clip(0)
            .astype("int64")
        )

    if "stock_quantity" in data:
        data["stock_quantity"] = (
            data["stock_quantity"]
            .clip(0)
            .astype("int64")
        )

    # ============================================================
    # 9. Một số giá trị float không được âm
    # ============================================================

    non_negative_float = [
        "customer_loyalty_score",
        "product_rating_avg",
        "unit_price_usd",
        "discount_amount_usd",
        "total_price_usd",
        "cost_usd",
        "tax_usd",
        "shipping_cost_usd",
        "session_duration_minutes"
    ]

    for col in non_negative_float:
        if col in data.columns:
            data[col] = data[col].clip(lower=0)

    if "fraud_risk_score" in data.columns:
        data["fraud_risk_score"] = (
            data["fraud_risk_score"]
            .clip(0, 1)
        )

    # ============================================================
    # 10. Tính lại các trường tài chính
    # ============================================================

    if all(
        col in data.columns
        for col in [
            "quantity",
            "unit_price_usd",
            "discount_percent"
        ]
    ):

        gross_amount = (
            data["quantity"] *
            data["unit_price_usd"]
        )

        data["discount_amount_usd"] = (
            gross_amount *
            data["discount_percent"] /
            100
        ).astype("float64")

        after_discount = (
            gross_amount -
            data["discount_amount_usd"]
        )

        # Thuế dựa trên dữ liệu gốc
        tax_rate = (
            source["tax_usd"] /
            (
                source["quantity"] *
                source["unit_price_usd"] -
                source["discount_amount_usd"]
            ).replace(0, np.nan)
        )

        tax_rate = tax_rate.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(tax_rate) > 0:
            sampled_tax_rate = rng.choice(
                tax_rate.values,
                size=n_rows,
                replace=True
            )

            data["tax_usd"] = (
                after_discount *
                sampled_tax_rate
            ).clip(lower=0)

        # Shipping
        if "shipping_cost_usd" in data.columns:
            data["shipping_cost_usd"] = (
                data["shipping_cost_usd"]
                .clip(lower=0)
            )

        # Tổng tiền
        data["total_price_usd"] = (
            after_discount +
            data["tax_usd"] +
            data["shipping_cost_usd"]
        ).astype("float64")

        # Cost
        cost_ratio = (
            source["cost_usd"] /
            source["total_price_usd"].replace(0, np.nan)
        )

        cost_ratio = cost_ratio.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(cost_ratio) > 0:
            sampled_cost_ratio = rng.choice(
                cost_ratio.values,
                size=n_rows,
                replace=True
            )

            data["cost_usd"] = (
                data["total_price_usd"] *
                sampled_cost_ratio
            ).clip(lower=0)

        # Profit
        data["profit_usd"] = (
            data["total_price_usd"] -
            data["cost_usd"]
        ).astype("float64")

        # Profit margin
        data["profit_margin_percent"] = np.where(
            data["total_price_usd"] != 0,
            data["profit_usd"] /
            data["total_price_usd"] * 100,
            0
        ).astype("float64")

    # ============================================================
    # 11. Ép lại kiểu dữ liệu cuối cùng
    # ============================================================

    for col in integer_columns:
        if col in data.columns:
            data[col] = (
                pd.to_numeric(data[col], errors="coerce")
                .fillna(0)
                .round()
                .astype("int64")
            )

    for col in float_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            ).astype("float64")

    data["order_date"] = pd.to_datetime(
        data["order_date"],
        errors="coerce"
    )

    data["account_creation_date"] = pd.to_datetime(
        data["account_creation_date"],
        errors="coerce"
    )

    return data



def main():
    parser = argparse.ArgumentParser(
        description="Sinh dữ liệu nguồn theo batch và chia thành các folder."
    )
    parser.add_argument("--input", default=r".\online_retail_II.xlsx",
                        help="Đường dẫn file CSV hoặc Excel nguồn")
    parser.add_argument("--sheet", default=None,
                        help="Tên sheet Excel cần dùng; mặc định ghép tất cả sheet")
    parser.add_argument("--output", default=r".\online_retail_500m_preprocessed",
                        help="Thư mục đầu ra")
    parser.add_argument("--target", type=int, default=500_000_000,
                        help="Tổng số dòng cần sinh")
    parser.add_argument("--rows-per-folder", type=int, default=100_000_000,
                        help="Số dòng mỗi folder")
    parser.add_argument("--batch-size", type=int, default=250_000,
                        help="Số dòng mỗi file Parquet (250000 dòng tạo 400 file mỗi folder 100M)")
    args = parser.parse_args()

    if args.target <= 0 or args.rows_per_folder <= 0 or args.batch_size <= 0:
        parser.error("--target, --rows-per-folder và --batch-size phải lớn hơn 0")
    if args.target % args.rows_per_folder != 0:
        parser.error("--target phải chia hết cho --rows-per-folder")
    if args.rows_per_folder % args.batch_size != 0:
        parser.error("--rows-per-folder phải chia hết cho --batch-size")

    input_file = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {input_file}")
    if output_root.exists() and any(output_root.iterdir()):
        parser.error(
            f"Thư mục đầu ra không rỗng, từ chối ghi đè dữ liệu: {output_root}"
        )

    log(f"Bắt đầu tiền xử lý; nguồn: {input_file}")
    stage_start = time.perf_counter()
    df = load_source(input_file, sheet_name=args.sheet)
    load_duration = time.perf_counter() - stage_start
    log(
        f"Đọc nguồn hoàn tất: {len(df):,} dòng, {len(df.columns)} cột; "
        f"thời lượng {format_duration(load_duration)}"
    )
    if df.empty:
        raise ValueError("File nguồn không có dữ liệu.")

    online_retail_columns = find_online_retail_columns(df)
    valid_retail_dates = None
    if online_retail_columns:
        stage_start = time.perf_counter()
        source_rows = len(df)
        df, removed_rows = preprocess_online_retail(df, online_retail_columns)
        preprocess_duration = time.perf_counter() - stage_start
        if df.empty:
            raise ValueError("Không còn dòng hợp lệ sau khi tiền xử lý.")
        valid_retail_dates = df["event_time"].to_numpy()
        log(
            f"Tiền xử lý Online Retail II hoàn tất: giữ {len(df):,}/{source_rows:,} "
            f"dòng, loại {removed_rows:,} dòng lỗi/không hợp lệ; "
            f"thời lượng {format_duration(preprocess_duration)}"
        )
        log(
            "Schema đầu ra: record_id, invoice_id, product_id, description, "
            "quantity, event_time, unit_price, customer_id, country, "
            "line_total, is_return"
        )
    else:
        log("Không nhận diện Online Retail II; dùng schema ecommerce hiện tại.")

    # Parse dates once before generation; the function still validates dates.
    if not online_retail_columns:
        for col in ("order_date", "account_creation_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    total_folders = args.target // args.rows_per_folder
    output_root.mkdir(parents=True, exist_ok=True)
    generated_total = 0
    start_order_id = 1
    global_part_idx = 0  # Đánh số file liên tục giữa các folder
    start_time = time.perf_counter()
    estimated_finish = None
    log(
        f"Bắt đầu sinh {args.target:,} dòng thành {total_folders} folder, "
        f"{args.batch_size:,} dòng/file tại {output_root}"
    )

    for folder_num in range(1, total_folders + 1):
        folder_path = output_root / f"folder_{folder_num}"
        folder_path.mkdir(parents=True, exist_ok=True)
        generated_folder = 0
        num_batches = args.rows_per_folder // args.batch_size
        folder_start = time.perf_counter()

        log(
            f"Bắt đầu folder {folder_num}/{total_folders}: "
            f"mục tiêu {args.rows_per_folder:,} dòng"
        )
        progress = tqdm(range(num_batches), desc=f"Folder {folder_num}", unit="file")
        for batch_idx in progress:
            current_size = min(args.batch_size, args.rows_per_folder - generated_folder)
            if online_retail_columns:
                chunk = generate_online_retail_chunk(
                    df,
                    current_size,
                    start_record_id=start_order_id,
                    valid_dates=valid_retail_dates,
                )
            else:
                chunk = generate_chunk(
                    df,
                    current_size,
                    start_order_id=start_order_id
                )

            output_path = folder_path / f"part_{global_part_idx:04d}.parquet"
            chunk.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)

            generated_folder += current_size
            generated_total += current_size
            start_order_id += current_size
            global_part_idx += 1
            elapsed = time.perf_counter() - start_time
            rate = generated_total / elapsed if elapsed > 0 else 0
            remaining = (
                (args.target - generated_total) / rate
                if rate > 0
                else 0
            )
            estimated_finish = datetime.now().astimezone() + timedelta(
                seconds=remaining
            )
            progress.set_postfix(
                rows=f"{generated_total:,}",
                rows_sec=f"{rate:,.0f}",
                eta=format_duration(remaining),
                finish=estimated_finish.strftime("%H:%M:%S"),
            )

            del chunk
            if (batch_idx + 1) % 10 == 0:
                gc.collect()

        log(
            f"Hoàn thành folder {folder_num}/{total_folders}: "
            f"{generated_folder:,} dòng; thời lượng "
            f"{format_duration(time.perf_counter() - folder_start)}"
        )

    elapsed = time.perf_counter() - start_time
    log(
        f"HOÀN THÀNH: {generated_total:,} dòng; tổng thời gian sinh "
        f"{format_duration(elapsed)}; ETA cuối cùng "
        f"{estimated_finish.strftime('%Y-%m-%d %H:%M:%S %Z') if estimated_finish else 'N/A'}; "
        f"thư mục đầu ra: {output_root}"
    )


if __name__ == "__main__":
    main()
