# sample_data/

This folder contains **auto-generated CSV files** produced by running `data/seed_data.py`.

> ⚠️ These files are excluded from git (see `.gitignore`).  
> To regenerate them, run: `python data/seed_data.py`

---

## Files

| File | Rows | Description |
|------|------|-------------|
| `orders.csv` | 3,600 | 180 days × 20 SKUs of daily demand history with channel & region |
| `inventory.csv` | 20 | Current on-hand, reserved, in-transit, reorder point, safety stock, EOQ per SKU |
| `suppliers.csv` | 5 | Vendor master — lead times, costs, reliability scores, contact info |

---

## Schema

### orders.csv
```
order_date, sku_id, quantity, channel, region
2024-01-01, SKU-001, 28.4, online, Northeast
```

### inventory.csv
```
sku_id, location_id, on_hand, reserved, in_transit, reorder_point, safety_stock, eoq, last_updated
SKU-001, DC-01, 412.0, 22.5, 0.0, 385.0, 350.0, 198.3, 2024-06-01T00:00:00
```

### suppliers.csv
```
supplier_id, name, country, lead_time_days, min_order_qty, unit_cost, reliability_score, status, contact_email
SUP-001, TechSource Global Ltd, China, 14, 50, 24.75, 0.97, preferred, orders@techsource.example.com
```
