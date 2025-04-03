"""
Clustering and Fitting Assignment
Student Name: Prardhan Mushke
Student ID: 24069965 
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Modern styling
sns.set_theme(style="whitegrid", palette="husl")
plt.style.use('ggplot')

def generate_sample_data():
    """Generate comprehensive realistic GDP sample data"""
    np.random.seed(42)  # For reproducible results
    countries = ['United States', 'China', 'United Kingdom', 
                'Germany', 'Japan', 'India', 'France', 'Brazil']
    
    # 2022 World Bank GDP per capita estimates (USD)
    base_gdp_2022 = {
        'United States': 76399, 
        'China': 12741,
        'United Kingdom': 45851,
        'Germany': 48432,
        'Japan': 33815,
        'India': 2389,
        'France': 43518,
        'Brazil': 8921
    }
    
    # Historical growth patterns (country-specific)
    growth_rates = {
        'United States': 1.018,
        'China': 1.072,
        'United Kingdom': 1.012,
        'Germany': 1.016,
        'Japan': 1.003,
        'India': 1.059,
        'France': 1.014,
        'Brazil': 1.011
    }
    
    years = range(2000, 2023)
    data = []
    
    for country in countries:
        for year in years:
            # Simulate realistic growth with some randomness
            annual_growth = growth_rates[country] * np.random.uniform(0.98, 1.02)
            years_from_2022 = year - 2022
            gdp = int(base_gdp_2022[country] * (annual_growth ** years_from_2022))
            
            data.append({
                'Country': country,
                'Year': year,
                'GDP_per_capita': gdp
            })
    
    return pd.DataFrame(data)

def load_data():
    """Load data with built-in sample generation"""
    print("Using enhanced World Bank-style sample data (2000-2022)")
    return generate_sample_data()

def create_relational_plot(data):
    """Line plot of GDP trends for selected countries"""
    plt.figure(figsize=(12, 6))
    
    # Select countries with most complete data
    plot_data = data.groupby('Country').filter(lambda x: len(x) == len(data['Year'].unique()))
    top_countries = plot_data['Country'].value_counts().nlargest(4).index
    
    if len(top_countries) == 0:
        print("Warning: No countries with complete time series")
        return
        
    sns.lineplot(data=plot_data[plot_data['Country'].isin(top_countries)], 
                x='Year', y='GDP_per_capita',
                hue='Country', style='Country', markers=True, linewidth=2.5)
    
    plt.title('GDP per Capita Trends (2000-2022)', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('GDP per Capita (USD)', fontsize=12)
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('relational_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_categorical_plot(data):
    """Bar plot of top countries by GDP"""
    plt.figure(figsize=(12, 6))
    latest_year = data['Year'].max()
    latest_data = data[data['Year'] == latest_year].nlargest(10, 'GDP_per_capita')
    
    if len(latest_data) == 0:
        print("Warning: No data for latest year")
        return
    
    # Fixed: Explicitly set hue to avoid warning
    ax = sns.barplot(data=latest_data, x='GDP_per_capita', y='Country',
                    hue='Country', palette='Blues_r', legend=False)
    
    # Add value labels
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 1000, p.get_y() + p.get_height()/2.,
                f'{width:,.0f}', ha='left', va='center')
    
    plt.title(f'Top 10 Economies by GDP per Capita ({latest_year})', fontsize=14)
    plt.xlabel('GDP per Capita (USD)', fontsize=12)
    plt.ylabel('')
    plt.xlim(0, latest_data['GDP_per_capita'].max() * 1.15)
    plt.tight_layout()
    plt.savefig('categorical_plot.png', dpi=300)
    plt.close()

def create_statistical_plot(data):
    """Boxplot distribution by year"""
    plt.figure(figsize=(14, 6))
    
    # Create decade bins for better visualization
    data['Decade'] = (data['Year'] // 10) * 10
    plot_data = data[data['Decade'].between(2000, 2020)]
    
    sns.boxplot(data=plot_data, x='Decade', y='GDP_per_capita',
               whis=[5, 95])  # Show 5th-95th percentiles
    
    plt.title('GDP Distribution by Decade (2000-2020)', fontsize=14)
    plt.xlabel('Decade', fontsize=12)
    plt.ylabel('GDP per Capita (USD)', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('statistical_plot.png', dpi=300)
    plt.close()

def perform_clustering(data):
    """K-means clustering on latest year's data with development categories"""
    latest_year = data['Year'].max()
    latest_data = data[data['Year'] == latest_year].copy()
    
    if len(latest_data) < 3:
        print("Skipping clustering: insufficient countries")
        return data
    
    # Create development categories
    latest_data['Development'] = pd.cut(latest_data['GDP_per_capita'],
                                      bins=[0, 5000, 25000, np.inf],
                                      labels=['Developing', 'Emerging', 'Developed'])
    
    # Prepare features
    X = latest_data[['GDP_per_capita']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dynamic cluster count
    n_clusters = min(3, len(latest_data))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    latest_data['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=latest_data, x='GDP_per_capita', y='Country',
                   hue='Development', style='Cluster',
                   palette='viridis', s=150)
    
    plt.title(f'Country Clusters by Development Stage ({latest_year})', fontsize=14)
    plt.xlabel('GDP per Capita (USD)', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('clustering_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return data

def perform_fitting(data):
    """Linear and polynomial fit for economic trends"""
    target_country = 'China'  # Most interesting growth story
    country_data = data[data['Country'] == target_country]
    
    if len(country_data) < 5:
        print(f"Skipping fitting: insufficient data for {target_country}")
        return None
    
    x = country_data['Year'].values
    y = country_data['GDP_per_capita'].values
    
    # Linear fit
    def linear_model(x, a, b):
        return a * x + b
    
    # Quadratic fit
    def growth_model(x, a, b, c):
        return a * x**2 + b * x + c
    
    try:
        # Fit both models
        lin_params, _ = curve_fit(linear_model, x, y)
        growth_params, _ = curve_fit(growth_model, x, y)
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        plt.scatter(x, y, label='Actual Data', s=80)
        
        x_cont = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_cont, linear_model(x_cont, *lin_params), 
                label=f'Linear Fit (R²={r2_score(y, linear_model(x, *lin_params)):.3f})')
        plt.plot(x_cont, growth_model(x_cont, *growth_params), 'g--',
                label=f'Growth Model (R²={r2_score(y, growth_model(x, *growth_params)):.3f})')
        
        plt.title(f'{target_country} GDP per Capita Growth (2000-2022)', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('GDP per Capita (USD)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('fitting_plot.png', dpi=300)
        plt.close()
        
        return {'linear': lin_params, 'growth': growth_params}
    
    except Exception as e:
        print(f"Fitting error: {str(e)}")
        return None

def analyze_statistical_moments(data):
    """Enhanced statistical analysis with disparity metrics"""
    latest_year = data['Year'].max()
    latest_data = data[data['Year'] == latest_year]
    
    if len(latest_data) < 2:
        print("Cannot calculate moments: insufficient data")
        return
    
    # Basic moments
    moments = pd.Series({
        'Mean': np.mean(latest_data['GDP_per_capita']),
        'Median': np.median(latest_data['GDP_per_capita']),
        'Variance': np.var(latest_data['GDP_per_capita']),
        'Skewness': ss.skew(latest_data['GDP_per_capita']),
        'Kurtosis': ss.kurtosis(latest_data['GDP_per_capita'])
    })
    
    # Disparity analysis
    top5 = latest_data.nlargest(5, 'GDP_per_capita')
    bottom5 = latest_data.nsmallest(5, 'GDP_per_capita')
    disparity_ratio = top5['GDP_per_capita'].mean() / bottom5['GDP_per_capita'].mean()
    
    print("\n=== Core Statistics ===")
    print(moments.to_string())
    
    print("\n=== Wealth Disparity ===")
    print(f"Top 5 Countries: {', '.join(top5['Country'].tolist())}")
    print(f"Bottom 5 Countries: {', '.join(bottom5['Country'].tolist())}")
    print(f"Wealth Ratio: {disparity_ratio:.1f}x (Top/Bottom)")
    
    print("\n=== Interpretation ===")
    print(f"1. Median GDP (${moments['Median']:,.0f}) vs Mean (${moments['Mean']:,.0f}) shows {'' if moments['Mean'] > moments['Median'] else 'no '}right-skew")
    print(f"2. Variance of {moments['Variance']:,.0f} indicates {'extreme' if moments['Variance'] > 1e8 else 'moderate'} wealth disparity")
    print(f"3. Skewness ({moments['Skewness']:.2f}): {'Right' if moments['Skewness'] > 0 else 'Left'}-skewed distribution")
    print(f"4. Kurtosis ({moments['Kurtosis']:.2f}): {'Heavy' if moments['Kurtosis'] > 0 else 'Light'}-tailed distribution")

def main():
    """Enhanced main workflow with progress tracking"""
    print("=== GDP Analysis Pipeline ===")
    print("1. Loading data...")
    data = load_data()
    print(f"   Loaded {len(data)} records ({data['Year'].min()}-{data['Year'].max()})")
    
    print("\n2. Creating visualizations...")
    create_relational_plot(data)
    create_categorical_plot(data)
    create_statistical_plot(data)
    print("   Saved: relational_plot.png, categorical_plot.png, statistical_plot.png")
    
    print("\n3. Running advanced analysis...")
    perform_clustering(data)
    perform_fitting(data)
    print("   Saved: clustering_plot.png, fitting_plot.png")
    
    print("\n4. Calculating statistics...")
    analyze_statistical_moments(data)
    
    print("\n=== Analysis Complete ===")
    print("All outputs saved to working directory")

if __name__ == "__main__":
    main()