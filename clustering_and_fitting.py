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

# Set consistent styling for all plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """
    Load and prepare GDP data for analysis.
    Returns a cleaned DataFrame with GDP per capita data.
    
    Returns:
        pd.DataFrame: Processed DataFrame with columns 'Country', 'Year', 'GDP_per_capita'
    """
    try:
        # Load data (in practice, use your data.csv or download as shown previously)
        df = pd.read_csv('data.csv')
        
        # Clean data - these steps will depend on your actual data structure
        df = df.melt(id_vars=['Country'], var_name='Year', value_name='GDP_per_capita')
        df['Year'] = pd.to_numeric(df['Year'])
        df = df.dropna()
        
        return df
    
    except FileNotFoundError:
        # Fallback to World Bank data if data.csv doesn't exist
        print("data.csv not found, using sample World Bank data")
        url = "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp-per-capita.csv"
        df = pd.read_csv(url)
        df.columns = ['Country', 'Year', 'GDP_per_capita']
        return df[df['Year'] >= 2000].dropna()

def create_relational_plot(data):
    """
    Create a relational plot (line plot) showing GDP trends.
    
    Args:
        data (pd.DataFrame): Processed DataFrame containing GDP data
    """
    plt.figure(figsize=(10, 6))
    
    # Select sample countries for clear visualization
    sample_countries = ['United States', 'China', 'United Kingdom', 'India']
    sample_data = data[data['Country'].isin(sample_countries)]
    
    sns.lineplot(data=sample_data, x='Year', y='GDP_per_capita', 
                 hue='Country', style='Country', markers=True)
    
    plt.title('GDP per Capita Trends (Line Plot)')
    plt.xlabel('Year')
    plt.ylabel('GDP per Capita (USD)')
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.show()

def create_categorical_plot(data):
    """
    Create a categorical plot (bar plot) of average GDP by country.
    
    Args:
        data (pd.DataFrame): Processed DataFrame containing GDP data
    """
    plt.figure(figsize=(10, 6))
    
    # Get latest year data and top 10 countries
    latest_year = data['Year'].max()
    latest_data = data[data['Year'] == latest_year]
    top_countries = latest_data.nlargest(10, 'GDP_per_capita')
    
    sns.barplot(data=top_countries, x='GDP_per_capita', y='Country')
    
    plt.title(f'Top 10 GDP per Capita by Country ({latest_year})')
    plt.xlabel('GDP per Capita (USD)')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.show()

def create_statistical_plot(data):
    """
    Create a statistical plot (box plot) showing GDP distribution.
    
    Args:
        data (pd.DataFrame): Processed DataFrame containing GDP data
    """
    plt.figure(figsize=(10, 6))
    
    # Get data from last 5 years
    recent_data = data[data['Year'] >= data['Year'].max() - 5]
    
    sns.boxplot(data=recent_data, x='Year', y='GDP_per_capita')
    
    plt.title('GDP per Capita Distribution (Box Plot)')
    plt.xlabel('Year')
    plt.ylabel('GDP per Capita (USD)')
    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.show()

def perform_clustering(data):
    """
    Perform K-means clustering on GDP data using 2 variables.
    
    Args:
        data (pd.DataFrame): Processed DataFrame containing GDP data
    
    Returns:
        np.array: Cluster labels for each data point
    """
    # Prepare data with exactly 2 variables as required
    cluster_data = data.pivot(index='Country', columns='Year', values='GDP_per_capita').dropna()
    X = cluster_data[[2010, 2020]].values  # Using only 2 years as required
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=clusters, palette='viridis')
    plt.title('Country Clustering by GDP (2010 vs 2020)')
    plt.xlabel('GDP per Capita 2010 (USD)')
    plt.ylabel('GDP per Capita 2020 (USD)')
    plt.tight_layout()
    plt.savefig('clustering_plot.png')
    plt.show()
    
    return clusters

def perform_fitting(data):
    """
    Perform curve fitting on GDP data using 1 feature and 1 target variable.
    
    Args:
        data (pd.DataFrame): Processed DataFrame containing GDP data
    
    Returns:
        tuple: Fitted parameters and covariance matrix
    """
    # Prepare data for one country (US) with Year as feature and GDP as target
    us_data = data[data['Country'] == 'United States'].sort_values('Year')
    x = us_data['Year'].values - 2000  # Using 1 feature (years since 2000)
    y = us_data['GDP_per_capita'].values  # Using 1 target (GDP)
    
    # Define and fit linear model
    def linear_model(x, a, b):
        return a * x + b
    
    params, cov = curve_fit(linear_model, x, y)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(x + 2000, y, label='Actual Data')
    plt.plot(x + 2000, linear_model(x, *params), 'r-', 
             label=f'Fit: y = {params[0]:.2f}x + {params[1]:.2f}')
    
    plt.title('US GDP per Capita with Linear Fit')
    plt.xlabel('Year')
    plt.ylabel('GDP per Capita (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fitting_plot.png')
    plt.show()
    
    return params, cov

def analyze_statistical_moments(data):
    """
    Calculate and display the four statistical moments for GDP data.
    
    Args:
        data (pd.DataFrame): Processed DataFrame containing GDP data
    """
    latest_data = data[data['Year'] == data['Year'].max()]['GDP_per_capita']
    
    moments = pd.Series({
        'Mean': np.mean(latest_data),
        'Variance': np.var(latest_data),
        'Skewness': ss.skew(latest_data),
        'Kurtosis': ss.kurtosis(latest_data)
    })
    
    print("\nStatistical Moments Analysis:")
    print(moments.to_string())
    
    print("\nInterpretation:")
    print("1. Mean: Central tendency of GDP values")
    print("2. Variance: Dispersion of GDP values")
    print("3. Skewness: Asymmetry of distribution (positive = right-skewed)")
    print("4. Kurtosis: Tail heaviness (positive = heavier tails than normal)")

def main():
    """
    Main function to execute all analysis steps.
    """
    print("Starting clustering and fitting analysis...")
    
    # Load and prepare data
    data = load_and_prepare_data()
    print(f"\nData loaded successfully. Contains {len(data)} records.")
    
    # Create required plots
    create_relational_plot(data)
    create_categorical_plot(data)
    create_statistical_plot(data)
    
    # Perform advanced analyses
    clusters = perform_clustering(data)
    params, cov = perform_fitting(data)
    analyze_statistical_moments(data)
    
    print("\nAnalysis completed successfully.")

if __name__ == "__main__":
    main()
