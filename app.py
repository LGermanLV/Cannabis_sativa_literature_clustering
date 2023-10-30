import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# Function to split text into lines of 12 words each
def split_abstract(abstract, words_per_line=12):
    if not isinstance(abstract, str):
        return "Not provided."
    elif len(abstract) == 0:
        return "Not provided."
    
    words = abstract.split()
    lines = [words[i:i + words_per_line] for i in range(0, len(words), words_per_line)]
    return '<br>'.join([' '.join(line) for line in lines])

# Load the CSV data from the provided link
data_url = "https://raw.githubusercontent.com/LGermanLV/Cannabis_sativa_literature_clustering/main/cannabis_df_final.csv"
new_df = pd.read_csv(data_url)

# Create the scatter plot
scatter_fig = go.Figure()
cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#fe4b4c', '#9467bd',
                  '#8c564b', '#a23f84', '#0f0f52', '#bcbd22', '#225a5f',
                  '#1cbfc1', '#50f600', '#07ffb4', '#eb42e7', '#7ad57a',
                  '#8b0404', '#a07fbf', '#ff917b', '#64815c', '#35248c',
                  '#6a6b00', '#00484f', '#0000ff']  # Define custom colors
for i, cluster in enumerate(new_df['y'].unique()):
    data = new_df[new_df['y'] == cluster]
    scatter_fig.add_trace(
        go.Scattergl(
            x=data['X_embedded_x'],
            y=data['X_embedded_y'],
            mode='markers',
            name=f'Cluster: {cluster}',
            marker=dict(size=5, opacity=0.7, color=cluster_colors[i]),  # Assign custom colors
            customdata=data[['Title', 'Authors', 'Journal', 'DOI', 'Year', 'Abstract']],
            text=[f'Title: {title}<br>Author(s): {authors}<br>Journal: {journal}<br>Link: {doi}<br>Abstract: {split_abstract(abstract)}' for title, authors, journal, doi, year, abstract in zip(data['Title'], data['Authors'], data['Journal'], data['DOI'], data['Year'], data['Abstract'])]
        )
    )

# Create the bar chart for cluster lengths
cluster_lengths = new_df['y'].value_counts()
bar_fig_cluster_lengths = go.Figure()
bar_fig_cluster_lengths.add_trace(go.Bar(x=cluster_lengths.index, y=cluster_lengths.values, marker_color=cluster_colors))
bar_fig_cluster_lengths.update_layout(
    #title="Cluster Lengths",
    xaxis_title="Cluster",
    yaxis_title="Number of Articles"
)

# Create the bar chart for articles per year with custom colors
articles_per_year = new_df['Year'].value_counts().sort_index()
custom_color_palette = px.colors.qualitative.Set1  # Define a custom color palette
colors = [custom_color_palette[i % len(custom_color_palette)] for i in range(len(articles_per_year))]
bar_fig_articles_per_year = go.Figure()
bar_fig_articles_per_year.add_trace(go.Bar(x=articles_per_year.index, y=articles_per_year.values, marker_color=colors))
bar_fig_articles_per_year.update_layout(
    #title="Articles per Year",
    xaxis_title="Year",
    yaxis_title="Amount of Articles"
)

# Create graphs for the number of articles per cluster per year
graphs_per_cluster_year = []
for cluster in new_df['y'].unique():
    cluster_data = new_df[new_df['y'] == cluster]
    articles_per_cluster_year = cluster_data['Year'].value_counts().sort_index()
    cluster_graph = go.Figure()
    cluster_graph.add_trace(go.Bar(x=articles_per_cluster_year.index, y=articles_per_cluster_year.values, marker_color=colors))
    cluster_graph.update_layout(
        title=f"Articles per Year (Cluster {cluster})",
        xaxis_title="Year",
        yaxis_title="Amount of Articles"
    )
    graphs_per_cluster_year.append(cluster_graph)

# Create graphs for the range in years covered by articles within each cluster
cluster_year_range_graphs = []
for cluster in new_df['y'].unique():
    cluster_data = new_df[new_df['y'] == cluster]
    min_year = cluster_data['Year'].min()
    max_year = cluster_data['Year'].max()
    cluster_year_range_graph = go.Figure()
    cluster_year_range_graph.add_trace(
        go.Scatter(
            x=[min_year, max_year],
            y=[1, 1],
            mode='markers+lines',
            marker=dict(size=10),
            line=dict(width=3),
            customdata=[min_year, max_year],
            text=f"Year Range: {min_year} - {max_year}"
        )
    )
    cluster_year_range_graph.update_layout(
        title=f"Year Range (Cluster {cluster})",
        xaxis_title="Year",
        yaxis_title="",
        xaxis=dict(range=[min_year - 1, max_year + 1])
    )
    cluster_year_range_graphs.append(cluster_year_range_graph)

# Create a list to hold the individual bar charts for each cluster
cluster_bar_charts = []

for cluster in new_df['y'].unique():
    cluster_data = new_df[new_df['y'] == cluster]
    articles_per_year = cluster_data['Year'].value_counts().sort_index().reset_index()
    articles_per_year.columns = ['Year', 'Number of Articles']

    # Create a bar chart for the cluster
    bar_chart = px.bar(articles_per_year, x='Number of Articles', y='Year', orientation='h',
                       title=f"Cluster {cluster} - Articles per Year",
                       labels={'Year': 'Year', 'Number of Articles': 'Number of Articles'})

    cluster_bar_charts.append(bar_chart)

# Combine all the cluster year range graphs into a single figure
combined_cluster_year_range_graph = go.Figure()

for cluster_year_range_graph in cluster_year_range_graphs:
    scatter = cluster_year_range_graph['data'][0]  # Extract the scatter data
    combined_cluster_year_range_graph.add_trace(scatter)

# Customize the layout for the combined graph
combined_cluster_year_range_graph.update_layout(
    title="Year Range Covered by Clusters",
    xaxis_title="Year",
    yaxis_title="",
)

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the app
app.layout = html.Div([
    html.H1("Clustering of the Cannabis Sativa Literature with t-SNE and K-Means", style={'font-weight': 'bold'}),
    html.P("This figure depicts the clustering patterns emerged upon applying an unsupervised machine learning (ML) technique called natural language processing (NLP) to a comprehensive dataset of 5480 scientific publications on Cannabis sativa spanning from 1843 to 2024. Notably, documents with similarly themed abstracts tend to group together and are represented by nearby cluster points on the plot. By hovering over these cluster points, additional details such as 'Title', 'Authors', 'Journal', 'DOI', and 'Year' will be displayed. Furthermore, users may obtain detailed information about individual clusters by clicking twice on any right-side cluster point. The upper-right portion of the three plots shown here also offer alternative ways to interact with the data."),

    dcc.Graph(figure=scatter_fig),
    html.H1("Clusters Length", style={'font-weight': 'bold'}),
    html.P("Pass the cursor over the bar corresponding to the cluster of interest to view the number of articles it contains."),

    dcc.Graph(figure=bar_fig_cluster_lengths),
    html.H1("Articles per Year", style={'font-weight': 'bold'}),
    html.P("Pass the cursor over any bar to reveal the number of articles published per year associated with cannabis sativa. Additionally, you can interact with the plot by zooming in or out over any desired time period using the buttons provided."),
    dcc.Graph(figure=bar_fig_articles_per_year),

    html.H1("Year Range Covered by Clusters", style={'font-weight': 'bold'}),
    html.P("These graphs show the range in years covered by articles within each cluster."),
    dcc.Graph(figure=combined_cluster_year_range_graph),

    html.H1("Distribution of Articles per Year in Clusters", style={'font-weight': 'bold'}),
    html.P("These bar charts illustrate the distribution of articles per year in each cluster."),
])

# Append the bar charts for each cluster
for bar_chart in cluster_bar_charts:
    app.layout.children.append(dcc.Graph(figure=bar_chart))

if __name__ == '__main__':
    app.run_server(debug=True)
