import pandas as pd
import matplotlib.pyplot as plt
import os


def main():
    data_file = os.path.join("..", "data", "orders.csv")

    print("--- 1. Loading Data & Converting Types ---")
    # CSV Load
    df = pd.read_csv(data_file)

    # Convert 'OrderDate' to datetime objects
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    print(df.dtypes)
    print("\n")

    print("--- 2. Adding TotalAmount Column ---")
    df['TotalAmount'] = df['Quantity'] * df['Price']
    print(df.head())
    print("\n")

    print("--- 3a. Total Store Income ---")
    total_income = df['TotalAmount'].sum()
    print(f"Total Income: ${total_income}")

    print("\n--- 3b. Average TotalAmount ---")
    avg_amount = df['TotalAmount'].mean()
    print(f"Average Order Amount: ${avg_amount:.2f}")

    print("\n--- 3c. Number of Orders per Customer ---")
    # value_counts() counts how many times each customer's name appears
    orders_per_customer = df['Customer'].value_counts()
    print(orders_per_customer)
    print("\n")

    print("--- 4. Orders with TotalAmount > 500 ---")
    high_value_orders = df[df['TotalAmount'] > 500]
    print(high_value_orders)
    print("\n")

    print("--- 5. Sort by OrderDate (Descending) ---")
    sorted_df = df.sort_values(by='OrderDate', ascending=False)
    print(sorted_df[['OrderID', 'OrderDate', 'TotalAmount']])
    print("\n")

    print("--- 6. Orders between June 5 and June 10 (Inclusive) ---")
    # YYYY-MM-DD string formats for datetime filtering
    mask = (df['OrderDate'] >= '2023-06-05') & (df['OrderDate'] <= '2023-06-10')
    june_subset = df[mask]
    print(june_subset[['OrderID', 'OrderDate']])
    print("\n")

    print("--- 7. Grouping by Category (*Extra) ---")
    # Using .agg() allows me to calculate multiple different stats at once
    category_stats = df.groupby('Category').agg(
        TotalItems=('Quantity', 'sum'),
        TotalSales=('TotalAmount', 'sum')
    )
    print(category_stats)
    print("\n")

    print("--- 8. TOP-3 Customers by TotalAmount ---")
    top_customers = df.groupby('Customer')['TotalAmount'].sum().nlargest(3)
    print(top_customers)
    print("\n")

    # --- Task 2: Data Visualization ---
    print("--- Bonus: Generating Charts ---")

    # 1. Line chart
    orders_by_date = df.groupby('OrderDate').size()

    plt.figure(figsize=(10, 5))
    orders_by_date.plot(kind='line', marker='o', title='Number of Orders by Date')
    plt.xlabel('Date')
    plt.ylabel('Number of Orders')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('orders_by_date.png')  # Saves the plot as an image
    plt.show()  # Opens the window to view it

    # 2. Bar chart
    income_by_category = df.groupby('Category')['TotalAmount'].sum()

    plt.figure(figsize=(8, 5))
    income_by_category.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'],
                            title='Income Distribution by Category')
    plt.xlabel('Category')
    plt.ylabel('Total Income ($)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('income_by_category.png')
    plt.show()


if __name__ == "__main__":
    main()