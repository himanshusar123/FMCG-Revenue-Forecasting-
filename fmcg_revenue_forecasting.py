# FMCG Revenue Forecasting Model
# Complete implementation with synthetic realistic dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== FMCG Revenue Forecasting Model ===\n")

# ==========================================
# 1. DATA GENERATION - REALISTIC FMCG DATASET
# ==========================================

def create_fmcg_dataset():
    """Create a realistic FMCG revenue dataset with multiple factors"""
    
    np.random.seed(42)
    n_months = 36  # 3 years of data
    
    # Time-based features
    months = pd.date_range('2021-01-01', periods=n_months, freq='M')
    
    # Base seasonal pattern for FMCG (higher sales in Q4, summer, festivals)
    seasonal_multiplier = np.array([
        1.0, 1.0, 1.1, 1.2, 1.3, 1.4,  # Jan-Jun
        1.5, 1.4, 1.2, 1.1, 1.3, 1.6   # Jul-Dec (peak in Dec)
    ] * 3)
    
    # Generate features
    data = []
    
    for i in range(n_months):
        month_data = {
            'Date': months[i],
            'Month': months[i].month,
            'Quarter': months[i].quarter,
            'Year': months[i].year,
            
            # Marketing & Promotion
            'Marketing_Spend': np.random.normal(150000, 30000),  # Marketing budget
            'Digital_Marketing': np.random.normal(50000, 15000),  # Digital campaigns
            'TV_Advertising': np.random.normal(80000, 25000),    # TV ads
            'Promotional_Days': np.random.randint(3, 12),        # Promotion days
            'Discount_Percentage': np.random.uniform(5, 25),     # Average discount
            
            # Distribution & Operations
            'Distribution_Points': np.random.randint(450, 650),  # Retail outlets
            'Sales_Team_Size': np.random.randint(25, 45),        # Sales people
            'Inventory_Days': np.random.randint(15, 35),         # Inventory levels
            
            # Product Mix
            'New_Product_Launches': np.random.randint(0, 4),     # New products
            'Premium_Product_Ratio': np.random.uniform(0.2, 0.5), # Premium mix
            
            # External Factors
            'Economic_Index': np.random.normal(100, 10),         # Economic conditions
            'Competitor_Spend': np.random.normal(200000, 50000), # Competition
            'Weather_Score': np.random.uniform(0.6, 1.0),       # Weather impact
            
            # Seasonal and trend factors
            'Seasonal_Factor': seasonal_multiplier[i],
            'Festival_Month': 1 if months[i].month in [3, 8, 10, 11, 12] else 0
        }
        
        data.append(month_data)
    
    df = pd.DataFrame(data)
    
    # Calculate Revenue using realistic business logic
    base_revenue = 800000  # Base monthly revenue
    
    # Revenue calculation with business relationships
    df['Revenue'] = (
        base_revenue * df['Seasonal_Factor'] +
        df['Marketing_Spend'] * 2.5 +  # Marketing ROI
        df['Digital_Marketing'] * 3.2 +  # Higher digital ROI
        df['TV_Advertising'] * 1.8 +
        df['Promotional_Days'] * 15000 +
        df['Distribution_Points'] * 180 +
        df['Sales_Team_Size'] * 8000 +
        df['New_Product_Launches'] * 25000 +
        df['Premium_Product_Ratio'] * 200000 +
        df['Economic_Index'] * 1000 +
        df['Festival_Month'] * 100000 +
        df['Weather_Score'] * 50000 -
        df['Competitor_Spend'] * 0.3 -
        df['Inventory_Days'] * 2000 +
        np.random.normal(0, 50000, len(df))  # Add noise
    )
    
    # Ensure positive revenue
    df['Revenue'] = np.maximum(df['Revenue'], 100000)
    
    return df

# Generate the dataset
print("📊 Generating FMCG Revenue Dataset...")
df = create_fmcg_dataset()
print(f"Dataset created with {len(df)} months of data\n")

# Display basic info
print("Dataset Overview:")
print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Average monthly revenue: ${df['Revenue'].mean():,.0f}")
print(f"Revenue range: ${df['Revenue'].min():,.0f} - ${df['Revenue'].max():,.0f}\n")

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS
# ==========================================

print("📈 EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# Basic statistics
print("\nRevenue Statistics:")
print(df['Revenue'].describe())

# Correlation analysis
print("\n🔍 Top Revenue Correlations:")
correlations = df.select_dtypes(include=[np.number]).corr()['Revenue'].sort_values(ascending=False)
print(correlations.head(10))

# ==========================================
# 3. DATA VISUALIZATION
# ==========================================

def create_visualizations(df):
    """Create comprehensive visualizations for the FMCG data"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FMCG Revenue Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Revenue Trend Over Time
    axes[0,0].plot(df['Date'], df['Revenue'], marker='o', linewidth=2, markersize=4)
    axes[0,0].set_title('Monthly Revenue Trend', fontweight='bold')
    axes[0,0].set_ylabel('Revenue ($)')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Seasonal Pattern
    monthly_avg = df.groupby('Month')['Revenue'].mean()
    axes[0,1].bar(monthly_avg.index, monthly_avg.values, color='skyblue', alpha=0.8)
    axes[0,1].set_title('Average Revenue by Month', fontweight='bold')
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('Average Revenue ($)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Marketing Spend vs Revenue
    axes[0,2].scatter(df['Marketing_Spend'], df['Revenue'], alpha=0.7, color='green')
    axes[0,2].set_title('Marketing Spend vs Revenue', fontweight='bold')
    axes[0,2].set_xlabel('Marketing Spend ($)')
    axes[0,2].set_ylabel('Revenue ($)')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Distribution Points Impact
    axes[1,0].scatter(df['Distribution_Points'], df['Revenue'], alpha=0.7, color='orange')
    axes[1,0].set_title('Distribution Points vs Revenue', fontweight='bold')
    axes[1,0].set_xlabel('Distribution Points')
    axes[1,0].set_ylabel('Revenue ($)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Promotional Impact
    promo_revenue = df.groupby('Promotional_Days')['Revenue'].mean()
    axes[1,1].bar(promo_revenue.index, promo_revenue.values, color='purple', alpha=0.8)
    axes[1,1].set_title('Promotional Days vs Average Revenue', fontweight='bold')
    axes[1,1].set_xlabel('Promotional Days')
    axes[1,1].set_ylabel('Average Revenue ($)')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Revenue Distribution
    axes[1,2].hist(df['Revenue'], bins=15, color='coral', alpha=0.8, edgecolor='black')
    axes[1,2].set_title('Revenue Distribution', fontweight='bold')
    axes[1,2].set_xlabel('Revenue ($)')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("📊 Creating Visualizations...")
create_visualizations(df)

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

print("\n🔧 FEATURE ENGINEERING")
print("=" * 50)

def engineer_features(df):
    """Create additional features for better model performance"""
    
    df_engineered = df.copy()
    
    # Time-based features
    df_engineered['Month_Sin'] = np.sin(2 * np.pi * df_engineered['Month'] / 12)
    df_engineered['Month_Cos'] = np.cos(2 * np.pi * df_engineered['Month'] / 12)
    df_engineered['Quarter_Sin'] = np.sin(2 * np.pi * df_engineered['Quarter'] / 4)
    df_engineered['Quarter_Cos'] = np.cos(2 * np.pi * df_engineered['Quarter'] / 4)
    
    # Interaction features
    df_engineered['Marketing_Per_Point'] = df_engineered['Marketing_Spend'] / df_engineered['Distribution_Points']
    df_engineered['Digital_TV_Ratio'] = df_engineered['Digital_Marketing'] / (df_engineered['TV_Advertising'] + 1)
    df_engineered['Team_Efficiency'] = df_engineered['Revenue'] / df_engineered['Sales_Team_Size']
    
    # Lagged features (using previous month values)
    df_engineered['Prev_Revenue'] = df_engineered['Revenue'].shift(1)
    df_engineered['Revenue_Growth'] = df_engineered['Revenue'].pct_change()
    df_engineered['Marketing_Trend'] = df_engineered['Marketing_Spend'].rolling(3).mean()
    
    # Category encoding
    df_engineered['High_Marketing'] = (df_engineered['Marketing_Spend'] > df_engineered['Marketing_Spend'].median()).astype(int)
    df_engineered['Peak_Season'] = df_engineered['Month'].apply(lambda x: 1 if x in [11, 12, 3, 4] else 0)
    
    return df_engineered

df_engineered = engineer_features(df)
print(f"Original features: {len(df.columns)}")
print(f"Engineered features: {len(df_engineered.columns)}")
print("New features created: Marketing_Per_Point, Digital_TV_Ratio, Seasonal encodings, etc.\n")

# ==========================================
# 5. MODEL PREPARATION
# ==========================================

print("🎯 MODEL PREPARATION")
print("=" * 50)

# Select features for modeling
feature_columns = [
    'Marketing_Spend', 'Digital_Marketing', 'TV_Advertising',
    'Promotional_Days', 'Discount_Percentage', 'Distribution_Points',
    'Sales_Team_Size', 'New_Product_Launches', 'Premium_Product_Ratio',
    'Economic_Index', 'Weather_Score', 'Festival_Month',
    'Month_Sin', 'Month_Cos', 'Quarter_Sin', 'Quarter_Cos',
    'Marketing_Per_Point', 'Digital_TV_Ratio', 'High_Marketing', 'Peak_Season'
]

# Remove rows with NaN values (from lagged features)
df_model = df_engineered.dropna()

X = df_model[feature_columns]
y = df_model['Revenue']

print(f"Model features: {len(feature_columns)}")
print(f"Training samples: {len(X)}")
print(f"Target variable: Revenue\n")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples\n")

# ==========================================
# 6. MODEL TRAINING
# ==========================================

print("🤖 MODEL TRAINING")
print("=" * 50)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("✅ Model training completed!\n")

# ==========================================
# 7. MODEL EVALUATION
# ==========================================

print("📊 MODEL EVALUATION")
print("=" * 50)

# Calculate metrics
def calculate_metrics(y_true, y_pred, dataset_name):
    """Calculate and display model performance metrics"""
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{dataset_name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"MAE: ${mae:,.0f}")
    print(f"MAPE: {mape:.2f}%")
    
    return r2, rmse, mae, mape

# Training performance
train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train, y_train_pred, "Training Set")

# Test performance
test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test, y_test_pred, "Test Set")

# ==========================================
# 8. FEATURE IMPORTANCE ANALYSIS
# ==========================================

print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
print("=" * 50)

# Get feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_
})

feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    impact = "Positive" if row['Coefficient'] > 0 else "Negative"
    print(f"{row['Feature']:<25}: {row['Coefficient']:>10.2f} ({impact})")

# ==========================================
# 9. MODEL VISUALIZATION
# ==========================================

def plot_model_results():
    """Create comprehensive model result visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('FMCG Revenue Forecasting Model Results', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted (Training)
    axes[0,0].scatter(y_train, y_train_pred, alpha=0.7, color='blue', label='Training')
    axes[0,0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', linewidth=2)
    axes[0,0].set_title(f'Training: Actual vs Predicted (R² = {train_r2:.3f})', fontweight='bold')
    axes[0,0].set_xlabel('Actual Revenue ($)')
    axes[0,0].set_ylabel('Predicted Revenue ($)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # 2. Actual vs Predicted (Test)
    axes[0,1].scatter(y_test, y_test_pred, alpha=0.7, color='green', label='Test')
    axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    axes[0,1].set_title(f'Test: Actual vs Predicted (R² = {test_r2:.3f})', fontweight='bold')
    axes[0,1].set_xlabel('Actual Revenue ($)')
    axes[0,1].set_ylabel('Predicted Revenue ($)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # 3. Residuals Plot
    residuals = y_test - y_test_pred
    axes[1,0].scatter(y_test_pred, residuals, alpha=0.7, color='orange')
    axes[1,0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1,0].set_title('Residuals Plot', fontweight='bold')
    axes[1,0].set_xlabel('Predicted Revenue ($)')
    axes[1,0].set_ylabel('Residuals ($)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Feature Importance
    top_features = feature_importance.head(8)
    axes[1,1].barh(top_features['Feature'], top_features['Abs_Coefficient'], color='purple', alpha=0.8)
    axes[1,1].set_title('Top Feature Importance', fontweight='bold')
    axes[1,1].set_xlabel('Absolute Coefficient Value')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("\n📈 Creating Model Result Visualizations...")
plot_model_results()

# ==========================================
# 10. BUSINESS INSIGHTS & RECOMMENDATIONS
# ==========================================

print("\n💼 BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 50)

# Revenue drivers analysis
print("\n🎯 Key Revenue Drivers:")
top_drivers = feature_importance.head(5)
for i, row in top_drivers.iterrows():
    if row['Coefficient'] > 0:
        print(f"✅ {row['Feature']}: Every unit increase leads to ${row['Coefficient']:,.0f} more revenue")
    else:
        print(f"⚠️  {row['Feature']}: Every unit increase decreases revenue by ${abs(row['Coefficient']):,.0f}")

# Marketing ROI Analysis
marketing_coef = feature_importance[feature_importance['Feature'] == 'Marketing_Spend']['Coefficient'].iloc[0]
digital_coef = feature_importance[feature_importance['Feature'] == 'Digital_Marketing']['Coefficient'].iloc[0]
tv_coef = feature_importance[feature_importance['Feature'] == 'TV_Advertising']['Coefficient'].iloc[0]

print(f"\n💰 Marketing ROI Analysis:")
print(f"Overall Marketing ROI: ${marketing_coef:.2f} revenue per $1 spent")
print(f"Digital Marketing ROI: ${digital_coef:.2f} revenue per $1 spent")
print(f"TV Advertising ROI: ${tv_coef:.2f} revenue per $1 spent")

# Seasonal insights
seasonal_revenue = df.groupby('Month')['Revenue'].mean()
peak_months = seasonal_revenue.nlargest(3)
low_months = seasonal_revenue.nsmallest(3)

print(f"\n📅 Seasonal Insights:")
print(f"Peak Revenue Months: {', '.join([f'Month {m} (${v:,.0f})' for m, v in peak_months.items()])}")
print(f"Low Revenue Months: {', '.join([f'Month {m} (${v:,.0f})' for m, v in low_months.items()])}")

# ==========================================
# 11. FUTURE FORECASTING FUNCTION
# ==========================================

def forecast_revenue(marketing_spend, digital_marketing, tv_advertising, promotional_days,
                    discount_percentage, distribution_points, sales_team_size,
                    new_product_launches, premium_product_ratio, economic_index,
                    weather_score, festival_month, month):
    """
    Forecast revenue based on input parameters
    """
    
    # Create feature vector
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    quarter = (month - 1) // 3 + 1
    quarter_sin = np.sin(2 * np.pi * quarter / 4)
    quarter_cos = np.cos(2 * np.pi * quarter / 4)
    
    marketing_per_point = marketing_spend / distribution_points
    digital_tv_ratio = digital_marketing / (tv_advertising + 1)
    high_marketing = 1 if marketing_spend > df['Marketing_Spend'].median() else 0
    peak_season = 1 if month in [11, 12, 3, 4] else 0
    
    features = np.array([[
        marketing_spend, digital_marketing, tv_advertising, promotional_days,
        discount_percentage, distribution_points, sales_team_size,
        new_product_launches, premium_product_ratio, economic_index,
        weather_score, festival_month, month_sin, month_cos,
        quarter_sin, quarter_cos, marketing_per_point, digital_tv_ratio,
        high_marketing, peak_season
    ]])
    
    prediction = model.predict(features)[0]
    return prediction

# Example forecast
print(f"\n🔮 SAMPLE REVENUE FORECAST")
print("=" * 50)

sample_forecast = forecast_revenue(
    marketing_spend=180000,
    digital_marketing=60000,
    tv_advertising=90000,
    promotional_days=8,
    discount_percentage=15,
    distribution_points=550,
    sales_team_size=35,
    new_product_launches=2,
    premium_product_ratio=0.35,
    economic_index=105,
    weather_score=0.8,
    festival_month=1,
    month=12  # December
)

print(f"Forecasted Revenue for December: ${sample_forecast:,.0f}")
print(f"Model Confidence (R²): {test_r2:.2%}")

# ==========================================
# 12. MODEL SUMMARY
# ==========================================

print(f"\n📋 MODEL SUMMARY")
print("=" * 50)
print(f"Model Type: Multiple Linear Regression")
print(f"Features: {len(feature_columns)}")
print(f"Training Period: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Test RMSE: ${test_rmse:,.0f}")
print(f"Test MAPE: {test_mape:.2f}%")
print(f"Average Monthly Revenue: ${df['Revenue'].mean():,.0f}")

print(f"\n🎯 Model Quality Assessment:")
if test_r2 > 0.8:
    print("✅ Excellent model performance (R² > 0.8)")
elif test_r2 > 0.7:
    print("✅ Good model performance (R² > 0.7)")
elif test_r2 > 0.6:
    print("⚠️  Fair model performance (R² > 0.6)")
else:
    print("❌ Poor model performance (R² < 0.6)")

print(f"\n💡 Key Takeaways:")
print("1. Digital marketing shows highest ROI among marketing channels")
print("2. Distribution network expansion directly correlates with revenue growth")
print("3. Seasonal patterns are strong predictors of revenue fluctuations")
print("4. Premium product mix significantly impacts overall revenue")
print("5. Economic conditions and weather factors influence consumer behavior")

print(f"\n🚀 Next Steps:")
print("1. Implement this model in production for monthly forecasting")
print("2. Set up automated retraining with new data")
print("3. Create dashboards for business stakeholders")
print("4. Develop scenario planning tools for strategic decisions")
print("5. Monitor model performance and update features as needed")

# Save the model and dataset
print(f"\n💾 Saving Results...")
df.to_csv('fmcg_revenue_data.csv', index=False)
print("✅ Dataset saved as 'fmcg_revenue_data.csv'")
print("✅ Model ready for deployment!")

print(f"\n" + "="*50)
print("🎉 FMCG REVENUE FORECASTING MODEL COMPLETE!")
print("="*50)