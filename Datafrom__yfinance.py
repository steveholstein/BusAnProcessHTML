from pathlib import Path
import yfinance as yf

# 1. Define output directory and file path
output_dir = Path(r"C:\Stevedrive\BusNewsAn\OutputSourcesforLLM")
output_dir.mkdir(parents=True, exist_ok=True)  # Creates directory if it doesn't exist

excel_path = output_dir / "asx200_monthly_history.xlsx"

# 2. Download ASX 200 data
asx200 = yf.download(
    tickers="^AXJO",
    start="2000-01-01",
    interval="1mo"  # Use '1d' for daily data
)

# 3. Flatten multi-index columns (common in newer yfinance versions)
if hasattr(asx200.columns, "levels") and len(asx200.columns.levels) > 1:
    asx200.columns = asx200.columns.get_level_values(0)

# 4. Save to Excel
# Note that the date written to file is the date at the start of each period;
# ie the date of the opening value

asx200.to_excel(excel_path, engine="openpyxl")

print(f"Saved successfully to: {excel_path}")
print(f"Rows written: {len(asx200)}")