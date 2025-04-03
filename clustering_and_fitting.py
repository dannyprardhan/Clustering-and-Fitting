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


# Configure visual settings
sns.set_theme(style="whitegrid", palette="husl")
plt.style.use("ggplot")


def generate_sample_data():
    """Generate realistic GDP sample data for demonstration."""
    np.random.seed(42)
    countries = [
        "United States", "China", "United Kingdom",
        "Germany", "Japan", "India", "France", "Brazil"
    ]
    
    base_gdp = {
        "United States": 76399, "China": 12741,
        "United Kingdom": 45851, "Germany": 48432,
        "Japan": 33815, "India": 2389,
        "France": 43518, "Brazil": 8921
    }
    
    growth_rates = {
        "United States": 1.018, "China": 1.072,
        "United Kingdom": 1.012, "Germany": 1.016,
        "Japan": 1.003, "India": 1.059,
        "France": 1.014, "Brazil": 1.011
    }

    years = range(2000, 2023)
    records = []
    
    for country in countries:
        for year in years:
            annual_growth = growth_rates[country] * np.random.uniform(0.98, 1.02)
            gdp = int(base_gdp[country] * (annual_growth ** (year - 2022)))
            records.append({
                "Country": country,
                "Year": year,
                "GDP_per_capita": gdp
            })
    
    return pd.DataFrame(records)


def load_data():
    """Load or generate sample data."""
    print("Loading World Bank-style sample data (2000-2022)")
    return generate_sample_data()


def create_relational_plot(data):
    """Visualize GDP trends over time."""
    plt.figure(figsize=(12, 6))
    
    complete_data = data.groupby("Country").filter(
        lambda x: len(x) == len(data["Year"].unique())
    )
    top_countries = complete_data["Country"].value_counts().nlargest(4).index
    
    if len(top_countries) > 0:
        sns.lineplot(
            data=complete_data[complete_data["Country"].isin(top_countries)],
            x="Year", y="GDP_per_capita",
            hue="Country", style="Country",
            markers=True, linewidth=2.5
        )
        
        plt.title("GDP per Capita Trends (2000-2022)", fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("GDP per Capita (USD)", fontsize=12)
        plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig("images/relational_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print("Warning: No countries with complete time series")


def perform_clustering(data):
    """Cluster countries by development level."""
    latest_data = data[data["Year"] == data["Year"].max()].copy()
    
    if len(latest_data) >= 3:
        latest_data["Development"] = pd.cut(
            latest_data["GDP_per_capita"],
            bins=[0, 5000, 25000, np.inf],
            labels=["Developing", "Emerging", "Developed"]
        )
        
        X = StandardScaler().fit_transform(latest_data[["GDP_per_capita"]])
        latest_data["Cluster"] = KMeans(n_clusters=3, random_state=42).fit_predict(X)
        
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            data=latest_data,
            x="GDP_per_capita", y="Country",
            hue="Development", style="Cluster",
            palette="viridis", s=150
        )
        
        plt.title("Country Development Clusters", fontsize=14)
        plt.tight_layout()
        plt.savefig("images/clustering_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    return data


def perform_fitting(data, country="China"):
    """Model GDP growth trends."""
    country_data = data[data["Country"] == country]
    
    if len(country_data) >= 5:
        x = country_data["Year"].values
        y = country_data["GDP_per_capita"].values
        
        def growth_model(x, a, b, c):
            return a * x**2 + b * x + c
        
        try:
            params, _ = curve_fit(growth_model, x, y)
            plt.figure(figsize=(12, 6))
            
            plt.scatter(x, y, label="Actual Data")
            x_range = np.linspace(x.min(), x.max(), 100)
            plt.plot(x_range, growth_model(x_range, *params), "r--",
                    label=f"Growth Model (R²={r2_score(y, growth_model(x, *params)):.3f})")
            
            plt.title(f"{country} GDP Growth Model", fontsize=14)
            plt.legend()
            plt.tight_layout()
            plt.savefig("images/fitting_plot.png", dpi=300)
            plt.close()
            
            return params
        except RuntimeError as e:
            print(f"Fitting failed: {str(e)}")
    return None


def main():
    """Execute analysis pipeline."""
    print("\n=== GDP Analysis Pipeline ===")
    data = load_data()
    
    print("\nGenerating visualizations...")
    create_relational_plot(data)
    perform_clustering(data)
    perform_fitting(data)
    
    print("\n=== Analysis Complete ===")
    print("Outputs saved to /images directory")


if __name__ == "__main__":
    main()