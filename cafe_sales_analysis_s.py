import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import streamlit as st

# ----------------------------
# Streamlit UI setup
# ----------------------------
st.set_page_config(page_title="Cafe Sales Analysis", layout="wide")
st.title("☕ Cafe Sales Data Cleaning & Analysis")
st.write("This app automatically cleans your cafe sales data and visualizes key insights.")

# ----------------------------
# Load CSV
# ----------------------------
input_path = Path("dirty_cafe_sales-1.csv")

if not input_path.exists():
    st.error("❌ Could not find 'dirty_cafe_sales-1.csv' in the current directory. Please add it to your repo.")
    st.stop()

na_tokens = ["", " ", "  ", "n/a", "na", "none", "null", "nan", "-", "--", "?", "missing", "unknown"]
df_raw = pd.read_csv(input_path, dtype=str, keep_default_na=False)

def clean_str_cell(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.lower() in na_tokens:
        return np.nan
    return s

df = df_raw.applymap(clean_str_cell)

# Normalize column names
def normalize_col(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"[\s_/]+", " ", col)
    col = col.replace("$", "").replace("#", "")
    return col

df.columns = [normalize_col(c) for c in df.columns]

# Map columns to canonical names
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

# Convert numeric columns
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

# Recalculate total spent where needed
calc_total = df["quantity"] * df["price per unit"]
tolerance = 0.01
needs_fix = df["total spent"].isna() | (np.abs(df["total spent"] - calc_total) > tolerance)
df.loc[needs_fix, "total spent"] = calc_total.loc[needs_fix]

# Clean string columns
for col in df.columns:
    if col not in ["quantity", "price per unit", "total spent"]:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        if col in ["item", "payment method"]:
            df[col] = df[col].apply(lambda x: x.title() if isinstance(x, str) else x)

# Create cleaned table
Tclean = df[df["total spent"].notna()].copy()

# Display preview
st.subheader("🧹 Cleaned Data Preview")
st.dataframe(Tclean.head())

# Summary stats
st.subheader("📊 Summary Statistics for Total Spent")
summary_stats = Tclean["total spent"].agg(
    Count="count",
    Mean="mean",
    Std="std",
    Min="min",
    Median="median",
    Max="max",
    Sum="sum",
).to_frame().T.round(2)
st.dataframe(summary_stats)

# Mostly sold items
item_counts = Tclean.dropna(subset=["item"]).groupby("item").size().sort_values(ascending=False)
qty_per_item = Tclean.groupby("item")["quantity"].sum().sort_values(ascending=False)
pay_counts = Tclean.dropna(subset=["payment method"]).groupby("payment method").size().sort_values(ascending=False)

st.subheader("🏆 Key Insights")
st.write(f"**Most Sold Item (by transactions):** {item_counts.index[0]} ({int(item_counts.iloc[0])})")
st.write(f"**Most Sold Item (by quantity):** {qty_per_item.index[0]} ({qty_per_item.iloc[0]:.0f})")
st.write(f"**Most Preferred Payment Method:** {pay_counts.index[0]} ({int(pay_counts.iloc[0])})")

# Visualization section
st.subheader("📈 Visualizations")

# Total Spent per Item
st.write("**Total Revenue per Item**")
fig1, ax1 = plt.subplots()
total_spent_per_item = Tclean.groupby("item")["total spent"].sum().sort_values(ascending=False)
ax1.bar(total_spent_per_item.index, total_spent_per_item.values)
ax1.set_title("Total Spent (Revenue) per Item")
ax1.set_xlabel("Item")
ax1.set_ylabel("Total Spent")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig1)

# Transactions per Item
st.write("**Number of Transactions per Item**")
fig2, ax2 = plt.subplots()
ax2.bar(item_counts.index, item_counts.values)
ax2.set_title("Number of Transactions per Item")
ax2.set_xlabel("Item")
ax2.set_ylabel("Transactions")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig2)

# Payment Method Pie Chart
if len(pay_counts) > 0:
    st.write("**Payment Method Distribution**")
    fig3, ax3 = plt.subplots()
    ax3.pie(pay_counts.values, labels=pay_counts.index, autopct="%1.1f%%", startangle=90)
    ax3.axis("equal")
    ax3.set_title("Payment Methods (Share of Transactions)")
    st.pyplot(fig3)

# Histogram of Total Spent
st.write("**Distribution of Total Spent per Transaction**")
fig4, ax4 = plt.subplots()
ax4.hist(Tclean["total spent"], bins=20)
ax4.set_title("Histogram of Total Spent per Transaction")
ax4.set_xlabel("Total Spent")
ax4.set_ylabel("Frequency")
st.pyplot(fig4)

st.success("✅ Analysis complete! All plots and summaries are displayed above.")