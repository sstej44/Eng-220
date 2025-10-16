import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

def clean_and_analyze_sales_data(input_file: str, output_file: str = "Tclean.csv"):
    # 1) Load the CSV and normalize
    na_tokens = ["", " ", "  ", "n/a", "na", "none", "null", "nan", "-", "--", "?", "missing", "unknown"]
    df_raw = pd.read_csv(input_file, dtype=str, keep_default_na=False)

    def clean_str_cell(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s.lower() in na_tokens:
            return np.nan
        return s

    df = df_raw.applymap(clean_str_cell)

    def normalize_col(col: str) -> str:
        col = col.strip().lower()
        col = re.sub(r"[\s_/]+", " ", col)
        col = col.replace("$", "").replace("#", "")
        return col

    df.columns = [normalize_col(c) for c in df.columns]

    col_variants = {
        "item": ["item", "product", "menu item", "item name"],
        "quantity": ["quantity", "qty", "count", "no of items", "number of items"],
        "price per unit": ["price per unit", "unit price", "price", "per unit price", "unit cost"],
        "total spent": ["total spent", "total", "amount", "line total", "revenue", "sale amount"],
        "payment method": ["payment method", "payment", "pay method", "payment type", "method"],
    }

    canonical_map = {}
    for canon, variants in col_variants.items():
        for v in variants:
            if v in df.columns:
                canonical_map[v] = canon
                break

    renamed_cols = {src: dst for src, dst in canonical_map.items()}
    df = df.rename(columns=renamed_cols)

    for col in ["item", "quantity", "price per unit", "total spent", "payment method"]:
        if col not in df.columns:
            df[col] = np.nan

    def to_numeric_series(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            cleaned = s.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
            cleaned = cleaned.str.replace(r"(?<=\d)\.(?=.*\.)", "", regex=True)
        else:
            cleaned = s
        return pd.to_numeric(cleaned, errors="coerce")

    df["quantity"] = to_numeric_series(df["quantity"])
    df["price per unit"] = to_numeric_series(df["price per unit"])
    df["total spent"] = to_numeric_series(df["total spent"])

    calc_total = df["quantity"] * df["price per unit"]
    tolerance = 0.01
    needs_fix = df["total spent"].isna() | (np.abs(df["total spent"] - calc_total) > tolerance)
    df.loc[needs_fix, "total spent"] = calc_total.loc[needs_fix]

    for col in df.columns:
        if col not in ["quantity", "price per unit", "total spent"]:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            if col in ["item", "payment method"]:
                df[col] = df[col].apply(lambda x: x.title() if isinstance(x, str) else x)

    Tclean = df[df["total spent"].notna()].copy()
    Tclean.to_csv(output_file, index=False)

    summary_stats = Tclean["total spent"].agg(
        Count="count",
        Mean="mean",
        Std="std",
        Min="min",
        Median="median",
        Max="max",
        Sum="sum",
    ).to_frame().T.round(2)

    item_counts = Tclean.dropna(subset=["item"]).groupby("item").size().sort_values(ascending=False)
    qty_per_item = Tclean.groupby("item")["quantity"].sum().sort_values(ascending=False)

    most_frequent_item = item_counts.index[0] if len(item_counts) else None
    greatest_quantity_item = qty_per_item.index[0] if len(qty_per_item) else None

    pay_counts = Tclean.dropna(subset=["payment method"]).groupby("payment method").size().sort_values(ascending=False)
    most_preferred_payment = pay_counts.index[0] if len(pay_counts) else None

    print("== Mostly Sold Item ==")
    print(f"By # of transactions: {most_frequent_item}")
    print(f"By total quantity: {greatest_quantity_item}")
    print("\n== Most Preferred Payment Method ==")
    print(f"{most_preferred_payment}")
    print(f"\nCleaned table saved to: {output_file}\n")

    total_spent_per_item = Tclean.groupby("item")["total spent"].sum().sort_values(ascending=False)
    plt.figure()
    total_spent_per_item.plot(kind="bar")
    plt.title("Total Spent (Revenue) per Item")
    plt.xlabel("Item")
    plt.ylabel("Total Spent")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure()
    item_counts.plot(kind="bar")
    plt.title("Number of Transactions per Item")
    plt.xlabel("Item")
    plt.ylabel("Transactions")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    if len(pay_counts) > 0:
        plt.figure()
        plt.pie(pay_counts.values, labels=pay_counts.index, autopct="%1.1f%%", startangle=90)
        plt.title("Payment Methods (Share of Transactions)")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    plt.figure()
    Tclean["total spent"].plot(kind="hist", bins=20)
    plt.title("Histogram of Total Spent per Transaction")
    plt.xlabel("Total Spent")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    return Tclean, summary_stats

if __name__ == "__main__":
    input_path = "dirty_cafe_sales-1.csv"
    clean_and_analyze_sales_data(input_path)
