import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── GENERATE DATASET ───────────────────────────────────────────────────────
n = 800
locations = np.random.choice(['Urban', 'Suburban', 'Rural'], n, p=[0.4, 0.4, 0.2])
area = np.where(locations == 'Urban',
                np.random.randint(500, 2500, n),
       np.where(locations == 'Suburban',
                np.random.randint(800, 4000, n),
                np.random.randint(1200, 6000, n)))
bedrooms = np.clip(np.random.poisson(3, n), 1, 7)
age = np.random.randint(0, 50, n)
bathrooms = np.clip(bedrooms - np.random.randint(0, 2, n), 1, 6)
garage = np.random.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])

# Price formula with location premium
loc_premium = np.where(locations == 'Urban', 1.5, np.where(locations == 'Suburban', 1.2, 1.0))
price = (area * 120 * loc_premium +
         bedrooms * 15000 +
         bathrooms * 8000 -
         age * 2000 +
         garage * 12000 +
         np.random.normal(0, 20000, n)).clip(50000)

df = pd.DataFrame({
    'area_sqft': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age_years': age,
    'garage_spaces': garage,
    'location': locations,
    'price': price.round(-3)
})
df.to_csv('data/house_prices.csv', index=False)
print("✅ Dataset saved")
print(df.head())

# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────
le = LabelEncoder()
df['location_encoded'] = le.fit_transform(df['location'])
df['price_per_sqft'] = df['price'] / df['area_sqft']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

features = ['area_sqft','bedrooms','bathrooms','age_years','garage_spaces','location_encoded','total_rooms']
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
print(f"\n📊 Results:  MSE={mse:,.0f}  RMSE={rmse:,.0f}  R²={r2:.4f}")

# ─── VISUALIZATIONS ─────────────────────────────────────────────────────────
plt.style.use('dark_background')
colors = ['#4ECDC4','#FF6B6B','#FFE66D','#A8E6CF','#FF8B94','#C3A6FF']

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#0A0E1A')
fig.suptitle('🏠 House Price Prediction Analytics', fontsize=22, fontweight='bold',
             color='white', y=1.01)

# 1. Price distribution
ax = axes[0, 0]
ax.hist(df['price']/1e6, bins=30, color=colors[0], edgecolor='#4ECDC4', alpha=0.8, linewidth=0.5)
ax.set_facecolor('#0F1923')
ax.set_title('💰 House Price Distribution', color='white', fontsize=13, pad=10)
ax.set_xlabel('Price (Million ₹)', color='#8B949E')
ax.set_ylabel('Count', color='#8B949E')
ax.tick_params(colors='#8B949E')
for spine in ax.spines.values(): spine.set_color('#1E2A3A')

# 2. Area vs Price
ax = axes[0, 1]
loc_colors = {'Urban': colors[1], 'Suburban': colors[0], 'Rural': colors[2]}
for loc in ['Urban', 'Suburban', 'Rural']:
    mask = df['location'] == loc
    ax.scatter(df[mask]['area_sqft'], df[mask]['price']/1e6,
               c=loc_colors[loc], alpha=0.5, s=15, label=loc, edgecolors='none')
ax.set_facecolor('#0F1923')
ax.set_title('📐 Area vs Price by Location', color='white', fontsize=13, pad=10)
ax.set_xlabel('Area (sqft)', color='#8B949E')
ax.set_ylabel('Price (Million ₹)', color='#8B949E')
ax.tick_params(colors='#8B949E')
for spine in ax.spines.values(): spine.set_color('#1E2A3A')
ax.legend(facecolor='#1E2A3A', edgecolor='#1E2A3A', labelcolor='white', fontsize=9)

# 3. Avg price by bedrooms
ax = axes[0, 2]
bed_price = df.groupby('bedrooms')['price'].mean() / 1e6
ax.bar(bed_price.index, bed_price.values, color=colors[5], alpha=0.85, edgecolor='#C3A6FF', linewidth=0.5)
ax.set_facecolor('#0F1923')
ax.set_title('🛏 Avg Price by Bedrooms', color='white', fontsize=13, pad=10)
ax.set_xlabel('Number of Bedrooms', color='#8B949E')
ax.set_ylabel('Avg Price (Million ₹)', color='#8B949E')
ax.tick_params(colors='#8B949E')
for spine in ax.spines.values(): spine.set_color('#1E2A3A')

# 4. Actual vs Predicted
ax = axes[1, 0]
ax.scatter(y_test/1e6, y_pred/1e6, alpha=0.5, color=colors[1], s=20, edgecolors='none')
mn, mx = (y_test/1e6).min(), (y_test/1e6).max()
ax.plot([mn, mx], [mn, mx], 'w--', linewidth=1.5, label='Perfect Prediction')
ax.set_facecolor('#0F1923')
ax.set_title('🎯 Actual vs Predicted Price', color='white', fontsize=13, pad=10)
ax.set_xlabel('Actual (Million ₹)', color='#8B949E')
ax.set_ylabel('Predicted (Million ₹)', color='#8B949E')
ax.tick_params(colors='#8B949E')
for spine in ax.spines.values(): spine.set_color('#1E2A3A')
ax.legend(facecolor='#1E2A3A', edgecolor='#1E2A3A', labelcolor='white', fontsize=9)
ax.text(0.05, 0.92, f'R² = {r2:.3f}', transform=ax.transAxes, color='#FFE66D',
        fontsize=11, fontweight='bold')

# 5. Feature importance (coefficients)
ax = axes[1, 1]
coeff = pd.Series(model.coef_, index=features).sort_values()
bar_colors = [colors[1] if v < 0 else colors[0] for v in coeff.values]
ax.barh(coeff.index, coeff.values, color=bar_colors, alpha=0.85)
ax.set_facecolor('#0F1923')
ax.set_title('🔍 Feature Coefficients', color='white', fontsize=13, pad=10)
ax.set_xlabel('Coefficient Value', color='#8B949E')
ax.tick_params(colors='#8B949E')
for spine in ax.spines.values(): spine.set_color('#1E2A3A')
ax.axvline(0, color='white', linewidth=0.8, alpha=0.5)

# 6. Age vs Price
ax = axes[1, 2]
sc = ax.scatter(df['age_years'], df['price']/1e6, c=df['area_sqft'],
                cmap='viridis', alpha=0.4, s=15, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax)
cbar.ax.tick_params(colors='#8B949E', labelsize=8)
cbar.set_label('Area (sqft)', color='#8B949E', fontsize=9)
ax.set_facecolor('#0F1923')
ax.set_title('🏗 Age vs Price (colored by Area)', color='white', fontsize=13, pad=10)
ax.set_xlabel('Age (years)', color='#8B949E')
ax.set_ylabel('Price (Million ₹)', color='#8B949E')
ax.tick_params(colors='#8B949E')
for spine in ax.spines.values(): spine.set_color('#1E2A3A')

plt.tight_layout(pad=2.5)
plt.savefig('outputs/house_price_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0A0E1A')
plt.close()
print("✅ Dashboard saved")

pd.Series({'MSE': round(mse), 'RMSE': round(rmse), 'R2': round(r2,4)}).to_csv('outputs/model_metrics.csv', header=False)
print("✅ Metrics saved")
