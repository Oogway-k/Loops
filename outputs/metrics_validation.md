# Chart value validation (python chart.py --check)

12 filled rows: 0 pass, 12 fail. Failed rows are marked in auto_check; chart.py flags them.

Checks: 1 source, 2 page, 3 support (figure on the metric's row or sentence, in the right year column; unit and consolidated scope match the page), 4 coverage.

| CSV line | Company | Metric | FY | Value | Unit | Report | PDF page | Result | Details | Human check |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | HCLTech | revenue | FY2025-26 | 130,144 | ₹ crore | HCLTech | 375 | fail | error: ModuleNotFoundError: No module named 'pymupdf' | - |
| 3 | HCLTech | net_profit | FY2025-26 | 16,652 | ₹ crore | HCLTech | 375 | fail | error: ModuleNotFoundError: No module named 'pymupdf' | - |
| 4 | HCLTech | headcount | FY2025-26 | 237,098 | employees | HCLTech | 216 | fail | error: ModuleNotFoundError: No module named 'pymupdf' | - |
| 5 | HCLTech | attrition | FY2025-26 | 12.51 | % | HCLTech | 217 | fail | error: ModuleNotFoundError: No module named 'pymupdf' | - |
| 6 | Infosys | revenue | FY2025-26 | 1,78,650 | ₹ crore | infosys-ar-26 | 287 | fail | error: ModuleNotFoundError: No module named 'pymupdf' | - |
| 7 | Infosys | net_profit | FY2025-26 | 29,474 | ₹ crore | infosys-ar-26 | 287 | fail | error: ModuleNotFoundError: No module named 'pymupdf' | - |
| 8 | Infosys | headcount | FY2025-26 | 3,52,095 | employees | infosys-ar-26 | 143 | fail | error: ModuleNotFoundError: No module named 'pymupdf' | - |
| 9 | Infosys | attrition | FY2025-26 | 12.6 | % | infosys-ar-26 | 144 | fail | error: ModuleNotFoundError: No module named 'pymupdf' | - |
| 10 | TCS | revenue | FY2025-26 | 2,67,021 | ₹ crore | TCS Annual Report 2025-26 | 167 | fail | error: ModuleNotFoundError: No module named 'pymupdf' | - |
| 11 | TCS | net_profit | FY2025-26 | 49,454 | ₹ crore | TCS Annual Report 2025-26 | 167 | fail | error: ModuleNotFoundError: No module named 'pymupdf' | - |
| 12 | TCS | headcount | FY2025-26 | 6,17,437 | employees | TCS Annual Report 2025-26 | 108 | fail | error: ModuleNotFoundError: No module named 'pymupdf' | - |
| 13 | TCS | attrition | FY2025-26 | 13.7 | % | TCS Annual Report 2025-26 | 108 | fail | error: ModuleNotFoundError: No module named 'pymupdf' | - |

## Coverage

- coverage: revenue has no passing value for HCLTech, Infosys, TCS
- coverage: net_profit has no passing value for HCLTech, Infosys, TCS
- coverage: headcount has no passing value for HCLTech, Infosys, TCS
- coverage: attrition has no passing value for HCLTech, Infosys, TCS
