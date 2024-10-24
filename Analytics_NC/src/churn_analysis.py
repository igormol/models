# churn_anaysis.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Function to compute churn scores
def churn_analysis_compute_churn_scores(data):
    frequency = data.groupby('Client_ID')['Sales_Date'].count()  # Count the number of sales per client
    recency = data.groupby('Client_ID')['Recency'].min()  # Get the minimum recency per client
    churn_data = pd.DataFrame({'Frequency': frequency, 'Recency': recency})  # Create a DataFrame with frequency and recency
    churn_data['Frequency'] = 1 / (1 + churn_data['Frequency'])  # Transform frequency to a score
    churn_data['Recency'] = churn_data['Recency'] / churn_data['Recency'].max()  # Normalize recency
    churn_data['Churn_Score'] = churn_data[['Frequency', 'Recency']].mean(axis=1)  # Compute the average churn score
    return churn_data

# Function to apply K-Means clustering and compute confidence scores
def churn_analysis_apply_kmeans_clustering(churn_data, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(churn_data[['Churn_Score']])  # Apply K-Means clustering
    churn_data['Cluster'] = kmeans.labels_  # Assign cluster labels to each client
    cluster_centers = kmeans.cluster_centers_  # Get the cluster centers

    # Compute confidence scores based on distances to cluster centroids
    distances = np.linalg.norm(churn_data[['Churn_Score']].values - cluster_centers[churn_data['Cluster']], axis=1)
    max_distance = np.max(distances)  # Find the maximum distance for normalization
    churn_data['Confidence'] = 1 - (distances / max_distance)  # Normalize distances to get confidence scores
    churn_data['Confidence'] = churn_data['Confidence'] * 100  # Normalize to percentage

    # Map cluster labels to descriptive names
    cluster_names = {0: 'Safe', 1: 'Low Risk', 2: 'Moderate Risk', 3: 'High Risk'}
    churn_data['Cluster_Name'] = churn_data['Cluster'].map(cluster_names)  # Map clusters to names
    return churn_data, kmeans

# Function to classify clients based on churn index intervals
def churn_analysis_classify_clients(churn_data):
    # Determine churn score intervals for each cluster
    intervals = {
        'Safe': (churn_data[churn_data['Cluster_Name'] == 'Safe']['Churn_Score'].min(), churn_data[churn_data['Cluster_Name'] == 'Safe']['Churn_Score'].max()),
        'Low Risk': (churn_data[churn_data['Cluster_Name'] == 'Low Risk']['Churn_Score'].min(), churn_data[churn_data['Cluster_Name'] == 'Low Risk']['Churn_Score'].max()),
        'Moderate Risk': (churn_data[churn_data['Cluster_Name'] == 'Moderate Risk']['Churn_Score'].min(), churn_data[churn_data['Cluster_Name'] == 'Moderate Risk']['Churn_Score'].max()),
        'High Risk': (churn_data[churn_data['Cluster_Name'] == 'High Risk']['Churn_Score'].min(), churn_data[churn_data['Cluster_Name'] == 'High Risk']['Churn_Score'].max())
    }
    return intervals

# Function to create and save the pie chart
def churn_analysis_create_pie_chart(churn_data, kmeans):
    cluster_counts = churn_data['Cluster_Name'].value_counts()  # Count the number of clients in each cluster
    average_churn = churn_data.groupby('Cluster_Name')['Churn_Score'].mean()  # Calculate average churn score per cluster
    average_confidence = churn_data.groupby('Cluster_Name')['Confidence'].mean()  # Calculate average confidence per cluster

    # Create labels for the pie chart
    labels = [f'{name}\n{count} clients\nAvg Churn: {churn:.2f}\nConfidence: {conf:.2f}'
              for name, count, churn, conf in zip(cluster_counts.index, cluster_counts.values, average_churn.values, average_confidence.values)]

    colors = ['#E07A5F', '#3D405B', '#81B29A', '#F2CC8F']  # Define color scheme

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(cluster_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.set_title('Client Churn Clusters')
    ax.legend(['Safe: Low Churn Score', 'Low Risk: Relatively Stable', 'Moderate Risk: Showing Signs of Churn', 'High Risk: High Churn Likelihood'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    return fig

# Function to create and save the bar chart
def churn_analysis_create_bar_chart(churn_data):
    cluster_counts = churn_data['Cluster_Name'].value_counts()  # Count the number of clients in each cluster
    average_churn = churn_data.groupby('Cluster_Name')['Churn_Score'].mean()  # Calculate average churn score per cluster
    average_confidence = churn_data.groupby('Cluster_Name')['Confidence'].mean()  # Calculate average confidence per cluster
    intervals = churn_analysis_classify_clients(churn_data)  # Get churn score intervals

    total_clients = len(churn_data)  # Get total number of clients
    bar_data = {
        'Cluster': [],
        'Count': [],
        'Percentage': [],
        'Avg_Churn': [],
        'Churn_Interval': [],
        'Avg_Confidence': []
    }

    # Populate bar chart data
    for cluster_name, count in cluster_counts.items():
        percentage = (count / total_clients) * 100
        avg_churn = average_churn[cluster_name]
        interval = intervals[cluster_name]
        avg_confidence = average_confidence[cluster_name]
        bar_data['Cluster'].append(cluster_name)
        bar_data['Count'].append(count)
        bar_data['Percentage'].append(percentage)
        bar_data['Avg_Churn'].append(avg_churn)
        bar_data['Churn_Interval'].append(f'{interval[0]:.2f} - {interval[1]:.2f}')
        bar_data['Avg_Confidence'].append(avg_confidence)

    df_bar = pd.DataFrame(bar_data)  # Convert to DataFrame
    colors = ['#E07A5F', '#3D405B', '#81B29A', '#F2CC8F']  # Define color scheme

    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(df_bar['Cluster'], df_bar['Percentage'], color=colors)
    for bar, perc, count, avg_churn, interval, avg_conf in zip(bars, df_bar['Percentage'], df_bar['Count'], df_bar['Avg_Churn'], df_bar['Churn_Interval'], df_bar['Avg_Confidence']):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{perc:.2f}%\n{count} clients\nAvg Churn: {avg_churn:.2f}\nChurn Interval: {interval}\nAvg Confidence: {avg_conf:.2f}%', ha='center', va='bottom')

    ax.set_title('Client Churn Clusters')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Percentage of Clients')
    ax.set_ylim(0, 100)  # Set y-axis range to 0-100%

    fig.tight_layout()  # Adjust layout to ensure everything fits well
    return fig

# Function to display and download table of results
def churn_analysis_show_table(churn_data):
    st.write("Table of Results")
    churn_data_sorted = churn_data.sort_values(by='Churn_Score', ascending=True)  # Sort data by churn score
    st.dataframe(churn_data_sorted[['Churn_Score', 'Confidence', 'Cluster_Name']])  # Display table

    # Download results table as CSV
    csv = churn_data_sorted[['Churn_Score', 'Confidence', 'Cluster_Name']].to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Table as CSV", data=csv, file_name='churn_results.csv', mime='text/csv')

# Function to display and download pie chart
def churn_analysis_show_pie_chart(churn_data, kmeans):
    st.write("Pie Chart")
    pie_chart = churn_analysis_create_pie_chart(churn_data, kmeans)  # Create pie chart
    st.pyplot(pie_chart)  # Display pie chart

    # Download pie chart as PNG
    buf = BytesIO()
    pie_chart.savefig(buf, format="png")
    st.download_button(label="Download Pie Chart", data=buf.getvalue(), file_name='pie_chart.png', mime='image/png')

# Function to display and download bar chart
def churn_analysis_show_bar_chart(churn_data):
    st.write("Bar Chart")
    bar_chart = churn_analysis_create_bar_chart(churn_data)  # Create bar chart
    st.pyplot(bar_chart)  # Display bar chart

    # Download bar chart as PNG
    buf = BytesIO()
    bar_chart.savefig(buf, format="png")
    st.download_button(label="Download Bar Chart", data=buf.getvalue(), file_name='bar_chart.png', mime='image/png')
