import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =====================================================================
# 1. REAL-WORLD DATASET GENERATION & INITIALIZATION
# =====================================================================
file_path = "retail_sales_data.csv"

if not os.path.exists(file_path):
    print(f"'{file_path}' not found. Generating real-world retail transactional data...")
    np.random.seed(101)
    
    # Generate 500 transactional records mimicking a retail point-of-sale system
    date_range = pd.date_range(start="2025-01-01", periods=100, freq="D")
    sample_data = {
        "Transaction_ID": range(10001, 10501),
        "Date": np.random.choice(date_range, size=500),
        "Product_Category": np.random.choice(["Electronics", "Clothing", "Home Goods", "Beauty"], size=500, p=[0.3, 0.4, 0.2, 0.1]),
        "Units_Sold": np.random.choice([1, 2, 3, 4, 5], size=500, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        "Unit_Price": np.random.uniform(10.0, 150.0, size=500),
        "Store_Location": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston"], size=500),
        "Customer_Segment": np.random.choice(["Member", "Regular"], size=500)
    }
    
    df_gen = pd.DataFrame(sample_data)
    # Derive Total Revenue column
    df_gen["Total_Revenue"] = df_gen["Units_Sold"] * df_gen["Unit_Price"]
    # Inject missing records to demonstrate data engineering resilience
    df_gen.loc[df_gen.sample(frac=0.04).index, "Unit_Price"] = np.nan
    df_gen["Total_Revenue"] = df_gen["Units_Sold"] * df_gen["Unit_Price"]
    
    df_gen.to_csv(file_path, index=False)

# Load dataset
df = pd.read_csv(file_path)

print("=== 1. DATA AUDIT & INSPECTION ===")
print(f"Total Transactions: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")
print("\nMissing values encountered before cleaning:")
print(df.isnull().sum())

# =====================================================================
# 2. DATA ENGINEERING & CLEANING
# =====================================================================
# Address missing numerical fields by computing the product median
if df["Unit_Price"].isnull().sum() > 0:
    median_price = df["Unit_Price"].median()
    df["Unit_Price"] = df["Unit_Price"].fillna(median_price)
    # Recalculate revenue based on fixed prices
    df["Total_Revenue"] = df["Units_Sold"] * df["Unit_Price"]

# Cast Date string objects into formal datetime structures
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.strftime("%Y-%m")

print("\nMissing values after cleaning pipeline:")
print(df.isnull().sum())

# =====================================================================
# 3. DOMAIN METRIC AGGREGATION & BUSINESS INSIGHTS
# =====================================================================
print("\n=== 2. EXECUTIVE BUSINESS REPORT ===")

total_sales_revenue = df["Total_Revenue"].sum()
average_basket_value = df["Total_Revenue"].mean()
total_units_moved = df["Units_Sold"].sum()

print(f"Gross Generated Revenue: ${total_sales_revenue:,.2f}")
print(f"Average Customer Ticket Value: ${average_basket_value:,.2f}")
print(f"Total Inventory Volume Sold: {total_units_moved} units")

# Category and Regional Performance
category_performance = df.groupby("Product_Category")["Total_Revenue"].sum().sort_values(ascending=False)
regional_performance = df.groupby("Store_Location")["Total_Revenue"].sum().sort_values(ascending=False)

print("\nRevenue Share By Department Category:")
print(category_performance.to_string())

print("\nRevenue Share By Store Location Geography:")
print(regional_performance.to_string())

# =====================================================================
# 4. DATA STORYTELLING VISUALIZATION EXECUTIVE DASHBOARD
# =====================================================================
print("\n=== 3. COMPILING EXECUTIVE DASHBOARD VISUALS ===")
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(16, 12))

# Subplot 1: Monthly Sales Revenue Trend Line
plt.subplot(2, 2, 1)
monthly_trend = df.groupby("Month")["Total_Revenue"].sum().reset_index()
sns.lineplot(data=monthly_trend, x="Month", y="Total_Revenue", marker="o", color="navy", linewidth=2.5)
plt.xticks(rotation=45)
plt.title("Revenue Trend Progression (Timeline Analysis)", fontsize=12, fontweight="bold")
plt.xlabel("Fiscal Month")
plt.ylabel("Gross Revenue ($)")

# Subplot 2: Revenue Matrix Partitioned by Category
plt.subplot(2, 2, 2)
sns.barplot(x=category_performance.values, y=category_performance.index, palette="Blues_r", hue=category_performance.index, legend=False)
plt.title("Department Category Contribution Analysis", fontsize=12, fontweight="bold")
plt.xlabel("Gross Revenue ($)")
plt.ylabel("Product Department")

# Subplot 3: Customer Behavior Breakdown (Member vs Regular)
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x="Customer_Segment", y="Total_Revenue", palette="Pastel1", hue="Customer_Segment", legend=False)
plt.title("Transaction Value Variance Across Loyalty Tiers", fontsize=12, fontweight="bold")
plt.xlabel("Customer Segment")
plt.ylabel("Transaction Total Revenue ($)")

# Subplot 4: Regional Market Share Comparison
plt.subplot(2, 2, 4)
plt.pie(regional_performance.values, labels=regional_performance.index, autopct="%1.1f%%", colors=["#2b5c8f", "#4682b4", "#6baed6", "#9ecae1"], startangle=140)
plt.title("Geographic Market Share Breakdown", fontsize=12, fontweight="bold")

plt.tight_layout()
print("Exhibiting final executive analytical dashboards.")
plt.show()
