import os
# Set Matplotlib config dir to workspace to avoid permission warning messages
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.matplotlib_cache')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_dataset(filepath, n_samples=200):
    """Generates a synthetic customer purchase behavior dataset."""
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Beauty', 'Sports & Outdoors']
    
    # Base columns
    age = np.random.randint(18, 70, size=n_samples)
    annual_income = np.random.randint(20, 150, size=n_samples)
    # Spending score ranges from 1 to 100
    spending_score = np.random.randint(1, 101, size=n_samples)
    satisfaction = np.random.randint(1, 6, size=n_samples)
    category = np.random.choice(categories, size=n_samples)
    
    # Introduce dependencies to make visualizations more meaningful
    # 1. Purchase amount correlates strongly with spending score and annual income
    purchase_amount = (spending_score * 3.8 + annual_income * 1.5 + np.random.normal(50, 30, size=n_samples)).round(2)
    # Ensure purchase amount stays positive and reasonable
    purchase_amount = np.clip(purchase_amount, 15.0, 850.0)
    
    # 2. Number of purchases correlates slightly with spending score and satisfaction
    num_purchases = (spending_score / 8 + satisfaction * 1.5 + np.random.randint(1, 8, size=n_samples)).astype(int)
    num_purchases = np.clip(num_purchases, 1, 30)

    data = {
        'CustomerID': [f'CUST-{i:04d}' for i in range(1, n_samples + 1)],
        'Age': age,
        'Annual Income (k$)': annual_income,
        'Spending Score (1-100)': spending_score,
        'Purchase Amount ($)': purchase_amount,
        'Number of Purchases': num_purchases,
        'Category': category,
        'Satisfaction Rating (1-5)': satisfaction
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"[Info] Synthetic dataset generated and saved to: {filepath}")
    return df

def perform_analysis(filepath):
    """Performs data cleaning, aggregation, and basic stats using Pandas."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        
    df = pd.read_csv(filepath)
    
    print("\n" + "="*60)
    print("               PANDAS DATA ANALYSIS REPORT")
    print("="*60)
    
    # 1. Shape of dataset
    print(f"Total Customers Analyzed: {df.shape[0]}")
    print(f"Attributes Per Customer: {df.shape[1]}")
    print("\n--- First 5 Records ---")
    print(df.head())
    
    # 2. Basic info
    print("\n--- Dataset Structure ---")
    print(df.info())
    
    # 3. Descriptive Stats
    print("\n--- Descriptive Statistics for Numeric Data ---")
    desc_stats = df.describe()
    print(desc_stats)
    
    # 4. Calculation of Averages (User's specific requirement)
    avg_age = df['Age'].mean()
    avg_income = df['Annual Income (k$)'].mean()
    avg_spending = df['Spending Score (1-100)'].mean()
    avg_purchase = df['Purchase Amount ($)'].mean()
    avg_purchases_num = df['Number of Purchases'].mean()
    avg_satisfaction = df['Satisfaction Rating (1-5)'].mean()
    
    print("\n--- Key Metrics (Average Values) ---")
    print(f"Average Customer Age:                 {avg_age:.2f} years")
    print(f"Average Annual Income:                ${avg_income:.2f}k")
    print(f"Average Spending Score (1-100):       {avg_spending:.2f}")
    print(f"Average Purchase Amount:              ${avg_purchase:.2f}")
    print(f"Average Number of Purchases:          {avg_purchases_num:.1f}")
    print(f"Average Customer Satisfaction Rating:  {avg_satisfaction:.2f} / 5.0")
    
    # 5. Grouped Aggregation
    print("\n--- Performance by Category ---")
    cat_summary = df.groupby('Category').agg(
        Total_Revenue=('Purchase Amount ($)', 'sum'),
        Average_Purchase_Amount=('Purchase Amount ($)', 'mean'),
        Average_Spending_Score=('Spending Score (1-100)', 'mean'),
        Customer_Count=('CustomerID', 'count')
    ).round(2).sort_values(by='Total_Revenue', ascending=False)
    print(cat_summary)
    
    print("="*60 + "\n")
    return df

def generate_visualizations(df, output_dirs):
    """Creates Bar Chart, Scatter Plot, and Heatmap and saves them to multiple directories."""
    # Ensure all output directories exist and are writable
    valid_dirs = []
    for d in output_dirs:
        try:
            os.makedirs(d, exist_ok=True)
            # Verify write access by writing a temporary file
            temp_path = os.path.join(d, '.write_test')
            with open(temp_path, 'w') as f:
                f.write('test')
            os.remove(temp_path)
            valid_dirs.append(d)
        except Exception as e:
            print(f"[Warning] Directory not writable, skipping: {d} (Error: {e})")
            
    if not valid_dirs:
        print("[Error] No writable output directories found. Cannot save visualizations.")
        return
        
    # Styling configuration
    sns.set_theme(style="whitegrid")
    
    # Define color scheme
    primary_color = "#3498db"
    accent_color = "#2ecc71"
    
    # ----------------------------------------------------
    # 1. Bar Chart: Average Purchase Amount by Product Category
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    category_data = df.groupby('Category')['Purchase Amount ($)'].mean().reset_index()
    category_data = category_data.sort_values(by='Purchase Amount ($)', ascending=False)
    
    # Palette based on categories
    colors = sns.color_palette("muted", len(category_data))
    
    bars = plt.bar(
        category_data['Category'], 
        category_data['Purchase Amount ($)'], 
        color=colors, 
        edgecolor='#7f8c8d', 
        linewidth=1,
        alpha=0.9
    )
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            height + 8, 
            f"${height:.2f}", 
            ha='center', 
            va='bottom', 
            fontsize=10, 
            fontweight='bold',
            color='#2c3e50'
        )
        
    plt.title('Average Purchase Value by Product Category', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Product Category', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Average Purchase ($)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylim(0, max(category_data['Purchase Amount ($)']) * 1.15)
    plt.tight_layout()
    
    # Save Bar Chart
    for d in valid_dirs:
        try:
            plt.savefig(os.path.join(d, 'bar_chart_category_purchase.png'), dpi=150)
        except Exception as e:
            print(f"[Error] Failed to save bar chart to {d}: {e}")
    plt.close()
    
    # ----------------------------------------------------
    # 2. Scatter Plot: Annual Income vs. Spending Score
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Bubble chart style: X=Income, Y=Spending Score, Hue=Category, Size=Purchase Amount
    scatter = sns.scatterplot(
        data=df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Category',
        size='Purchase Amount ($)',
        sizes=(40, 400),
        palette='Set1',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.title('Annual Income vs. Spending Score (by Category & Purchase Value)', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Annual Income (k$)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Spending Score (1-100)', fontsize=12, fontweight='bold', labelpad=10)
    
    # Position the legend outside
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    
    # Save Scatter Plot
    for d in valid_dirs:
        try:
            plt.savefig(os.path.join(d, 'scatter_plot_income_vs_spending.png'), dpi=150)
        except Exception as e:
            print(f"[Error] Failed to save scatter plot to {d}: {e}")
    plt.close()
    
    # ----------------------------------------------------
    # 3. Heatmap: Correlation of Numerical Features
    # ----------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    numerical_cols = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_cols.corr()
    
    # Mask to show only half of the symmetric matrix
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        fmt=".3f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8}
    )
    
    plt.title('Correlation Matrix of Customer Metrics', fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save Heatmap
    for d in valid_dirs:
        try:
            plt.savefig(os.path.join(d, 'heatmap_correlation.png'), dpi=150)
        except Exception as e:
            print(f"[Error] Failed to save heatmap to {d}: {e}")
    plt.close()
    
    print(f"[Info] Visualizations successfully saved to output directories.")

if __name__ == '__main__':
    csv_file = 'customer_behavior.csv'
    
    # Output directories: one in workspace, one in artifact directory for embedding
    artifact_dir = '/Users/harshal/.gemini/antigravity/brain/a0400bc8-2555-40a0-a48d-a49b463f5bbd'
    output_dirs = [
        os.path.join(os.getcwd(), 'visualizations'),
        os.path.join(artifact_dir, 'visualizations')
    ]
    
    # Generate, analyze, and visualize
    generate_dataset(csv_file)
    df = perform_analysis(csv_file)
    generate_visualizations(df, output_dirs)
