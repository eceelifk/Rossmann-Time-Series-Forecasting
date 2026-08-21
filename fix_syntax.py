import nbformat

with open('rossmann_sales_forecast.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "['Sales_Rolling_7_Scaled'] + state_holiday_cols = " in cell.source:
            # Fix the assignment error
            cell.source = cell.source.replace(
                "['Sales_Rolling_7_Scaled'] + state_holiday_cols = [c for c in df.columns if 'StateHoliday_' in c]",
                "state_holiday_cols = [c for c in df.columns if 'StateHoliday_' in c]"
            )

with open('rossmann_sales_forecast.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Syntax error fixed.")
