import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, callback, no_update
import dash_bootstrap_components as dbc
import os
import warnings
import json
from plotly.subplots import make_subplots
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
os.makedirs('output/dashboards', exist_ok=True)

try:
    # Load the data
    print("Loading data from data/onlinefoods.csv...")
    df = pd.read_csv('data/onlinefoods.csv')
    
    # Clean the data
    # Remove trailing commas from column names if present
    df.columns = df.columns.str.rstrip(',')
    
    # Convert 'Output' to boolean
    df['Output'] = df['Output'].map({'Yes': True, 'No': False})
    
    # Convert the last column to boolean if it exists and contains Yes/No
    if df.columns[-1].startswith('Unnamed') or df.columns[-1] == '':
        # Rename it to something meaningful
        df.rename(columns={df.columns[-1]: 'Uses_Online_Food'}, inplace=True)
        # Convert to boolean
        df['Uses_Online_Food'] = df['Uses_Online_Food'].map({'Yes': True, 'No': False})
    
    # Ensure all numeric columns are properly typed
    for col in ['Age', 'Family size', 'Pin code']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure latitude and longitude are float
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Fill any missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Family size'].fillna(df['Family size'].median(), inplace=True)
    df['Pin code'].fillna(df['Pin code'].mode()[0], inplace=True)
    df['latitude'].fillna(df['latitude'].mean(), inplace=True)
    df['longitude'].fillna(df['longitude'].mean(), inplace=True)
    
    # Categorical columns
    categorical_cols = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 
                        'Educational Qualifications', 'Feedback']
    
    # Fill missing values in categorical columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    print(traceback.format_exc())
    # Create a minimal dataframe for demonstration if data loading fails
    df = pd.DataFrame({
        'Age': np.random.randint(18, 60, 100),
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Marital Status': np.random.choice(['Single', 'Married'], 100),
        'Occupation': np.random.choice(['Student', 'Employee', 'Self Employed'], 100),
        'Monthly Income': np.random.choice(['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'], 100),
        'Educational Qualifications': np.random.choice(['Graduate', 'Post Graduate', 'Ph.D'], 100),
        'Family size': np.random.randint(1, 7, 100),
        'latitude': np.random.uniform(12.8, 13.1, 100),
        'longitude': np.random.uniform(77.5, 77.8, 100),
        'Pin code': np.random.randint(560001, 560100, 100),
        'Output': np.random.choice([True, False], 100),
        'Feedback': np.random.choice(['Positive', 'Negative'], 100)
    })
    print("Using demo data instead.")

# Initialize the Dash app with a Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Online Foods EDA Dashboard"

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Online Foods Exploratory Data Analysis Dashboard", 
                    className="text-center text-primary mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    html.P("Age Range:"),
                    dcc.RangeSlider(
                        id='age-slider',
                        min=int(df['Age'].min()),
                        max=int(df['Age'].max()),
                        value=[int(df['Age'].min()), int(df['Age'].max())],
                        marks={i: str(i) for i in range(int(df['Age'].min()), int(df['Age'].max())+1, 2)},
                        step=1
                    ),
                    html.Br(),
                    html.P("Gender:"),
                    dcc.Dropdown(
                        id='gender-dropdown',
                        options=[{'label': i, 'value': i} for i in sorted(df['Gender'].unique())],
                        value=sorted(df['Gender'].unique()),
                        multi=True
                    ),
                    html.Br(),
                    html.P("Marital Status:"),
                    dcc.Dropdown(
                        id='marital-dropdown',
                        options=[{'label': i, 'value': i} for i in sorted(df['Marital Status'].unique())],
                        value=sorted(df['Marital Status'].unique()),
                        multi=True
                    ),
                    html.Br(),
                    html.P("Occupation:"),
                    dcc.Dropdown(
                        id='occupation-dropdown',
                        options=[{'label': i, 'value': i} for i in sorted(df['Occupation'].unique())],
                        value=sorted(df['Occupation'].unique()),
                        multi=True
                    )
                ])
            ]),
            html.Div(id='filter-summary', className="mt-3")
        ], width=3),
        
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Demographics", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='age-histogram')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='gender-pie')
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='marital-bar')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='occupation-bar')
                        ], width=6)
                    ])
                ]),
                dbc.Tab(label="Socioeconomic Factors", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='income-bar')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='education-bar')
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='family-bar')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='income-education-heatmap')
                        ], width=6)
                    ])
                ]),
                dbc.Tab(label="Online Food Usage", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='output-pie')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='feedback-pie')
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='age-output-box')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='gender-output-bar')
                        ], width=6)
                    ])
                ]),
                dbc.Tab(label="Geographic Analysis", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='map-scatter')
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='pincode-bar')
                        ], width=12)
                    ])
                ]),
                dbc.Tab(label="Advanced Analysis", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='age-family-scatter')
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='sunburst-chart')
                        ], width=12)
                    ])
                ])
            ])
        ], width=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Key Insights", className="text-center text-primary mb-3"),
                html.Ul(id='insights-list')
            ], className="mt-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Footer([
                html.P("Online Foods EDA Dashboard © 2023", className="text-center text-muted mt-4")
            ])
        ], width=12)
    ])
], fluid=True)

# Add a callback to display filter summary
@app.callback(
    Output('filter-summary', 'children'),
    [Input('age-slider', 'value'),
     Input('gender-dropdown', 'value'),
     Input('marital-dropdown', 'value'),
     Input('occupation-dropdown', 'value')]
)
def update_filter_summary(age_range, genders, marital_statuses, occupations):
    try:
        # Create a filter summary card
        return dbc.Card([
            dbc.CardBody([
                html.H5("Current Filters", className="card-title"),
                html.P(f"Age: {age_range[0]} to {age_range[1]}"),
                html.P(f"Gender: {', '.join(genders) if genders else 'All'}"),
                html.P(f"Marital Status: {', '.join(marital_statuses) if marital_statuses else 'All'}"),
                html.P(f"Occupation: {', '.join(occupations[:3]) + '...' if len(occupations) > 3 else ', '.join(occupations) if occupations else 'All'}")
            ])
        ])
    except Exception as e:
        print(f"Error in filter summary: {e}")
        return dbc.Card([
            dbc.CardBody([
                html.H5("Current Filters", className="card-title"),
                html.P("Error displaying filters")
            ])
        ])

# Define callbacks for all visualizations
@app.callback(
    [Output('age-histogram', 'figure'),
     Output('gender-pie', 'figure'),
     Output('marital-bar', 'figure'),
     Output('occupation-bar', 'figure'),
     Output('income-bar', 'figure'),
     Output('education-bar', 'figure'),
     Output('family-bar', 'figure'),
     Output('income-education-heatmap', 'figure'),
     Output('output-pie', 'figure'),
     Output('feedback-pie', 'figure'),
     Output('age-output-box', 'figure'),
     Output('gender-output-bar', 'figure'),
     Output('map-scatter', 'figure'),
     Output('pincode-bar', 'figure'),
     Output('age-family-scatter', 'figure'),
     Output('sunburst-chart', 'figure'),
     Output('insights-list', 'children')],
    [Input('age-slider', 'value'),
     Input('gender-dropdown', 'value'),
     Input('marital-dropdown', 'value'),
     Input('occupation-dropdown', 'value')]
)
def update_graphs(age_range, genders, marital_statuses, occupations):
    try:
        # Filter the dataframe based on the selections
        filtered_df = df.copy()
        
        # Apply filters
        if age_range:
            filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
        
        if genders and len(genders) > 0:
            filtered_df = filtered_df[filtered_df['Gender'].isin(genders)]
        
        if marital_statuses and len(marital_statuses) > 0:
            filtered_df = filtered_df[filtered_df['Marital Status'].isin(marital_statuses)]
        
        if occupations and len(occupations) > 0:
            filtered_df = filtered_df[filtered_df['Occupation'].isin(occupations)]
        
        # Check if filtered dataframe is empty
        if filtered_df.empty:
            # Return empty figures with a message
            empty_fig = px.scatter(title="No data available for the selected filters")
            empty_fig.update_layout(
                annotations=[{
                    'text': 'No data available for the selected filters. Please adjust your filters.',
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
            return [empty_fig] * 16 + [[html.Li("No data available for the selected filters.")]]
        
        # Create the figures
        # Age histogram
        age_hist = px.histogram(filtered_df, x='Age', color='Gender', 
                               title='Age Distribution by Gender',
                               labels={'Age': 'Age', 'count': 'Count'},
                               marginal='box')
        
        # Gender pie chart
        gender_pie = px.pie(filtered_df, names='Gender', title='Gender Distribution',
                           hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
        
        # Marital status bar chart
        marital_counts = filtered_df['Marital Status'].value_counts().reset_index()
        marital_counts.columns = ['Marital Status', 'Count']
        marital_bar = px.bar(marital_counts, 
                            x='Marital Status', y='Count', title='Marital Status Distribution',
                            color='Marital Status', color_discrete_sequence=px.colors.qualitative.Pastel)
        
        # Occupation bar chart
        occupation_counts = filtered_df['Occupation'].value_counts().reset_index()
        occupation_counts.columns = ['Occupation', 'Count']
        occupation_bar = px.bar(occupation_counts, 
                               x='Occupation', y='Count', title='Occupation Distribution',
                               color='Occupation', color_discrete_sequence=px.colors.qualitative.Pastel)
        occupation_bar.update_layout(xaxis={'categoryorder': 'total descending'})
        
        # Income bar chart
        income_counts = filtered_df['Monthly Income'].value_counts().reset_index()
        income_counts.columns = ['Monthly Income', 'Count']
        income_bar = px.bar(income_counts, 
                           x='Monthly Income', y='Count', title='Monthly Income Distribution',
                           color='Monthly Income', color_discrete_sequence=px.colors.qualitative.Pastel)
        income_bar.update_layout(xaxis={'categoryorder': 'total descending'})
        
        # Education bar chart
        education_counts = filtered_df['Educational Qualifications'].value_counts().reset_index()
        education_counts.columns = ['Educational Qualifications', 'Count']
        education_bar = px.bar(education_counts, 
                              x='Educational Qualifications', y='Count', 
                              title='Educational Qualifications Distribution',
                              color='Educational Qualifications', 
                              color_discrete_sequence=px.colors.qualitative.Pastel)
        education_bar.update_layout(xaxis={'categoryorder': 'total descending'})
        
        # Family size bar chart
        family_counts = filtered_df['Family size'].value_counts().reset_index()
        family_counts.columns = ['Family Size', 'Count']
        family_bar = px.bar(family_counts,
                           x='Family Size', y='Count', title='Family Size Distribution',
                           color='Family Size', color_discrete_sequence=px.colors.qualitative.Pastel)
        
        # Income-Education heatmap
        try:
            income_edu_crosstab = pd.crosstab(filtered_df['Monthly Income'], 
                                             filtered_df['Educational Qualifications'])
            income_edu_heatmap = px.imshow(income_edu_crosstab, title='Income vs Education',
                                          labels=dict(x='Educational Qualifications', 
                                                     y='Monthly Income', color='Count'),
                                          color_continuous_scale='Viridis')
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            income_edu_heatmap = px.imshow(pd.DataFrame([[0]]), title='Income vs Education - Error')
            income_edu_heatmap.update_layout(
                annotations=[{
                    'text': 'Error creating heatmap. Insufficient data.',
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
        
        # Output pie chart
        # Create a DataFrame with counts to avoid boolean indexing issues
        output_counts = filtered_df['Output'].value_counts().reset_index()
        output_counts.columns = ['Output', 'Count']
        output_pie = px.pie(output_counts, names='Output', values='Count', 
                           title='Online Food Service Usage',
                           hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
        # Update labels
        output_pie.update_traces(
            textinfo='percent+label',
            text=output_counts['Output'].map({True: 'Yes', False: 'No'})
        )
        
        # Feedback pie chart
        feedback_counts = filtered_df['Feedback'].value_counts().reset_index()
        feedback_counts.columns = ['Feedback', 'Count']
        feedback_pie = px.pie(feedback_counts, names='Feedback', values='Count',
                             title='Feedback Distribution',
                             hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
        
        # Age vs Output box plot
        age_output_box = px.box(filtered_df, x='Output', y='Age', 
                               title='Age vs Online Food Service Usage',
                               color='Output', color_discrete_sequence=px.colors.qualitative.Pastel)
        # Update x-axis labels
        age_output_box.update_xaxes(
            ticktext=['Yes', 'No'],
            tickvals=[True, False]
        )
        
        # Gender vs Output bar chart
        try:
            # Create a crosstab with counts instead of percentages to avoid division issues
            gender_output_counts = pd.crosstab(filtered_df['Gender'], filtered_df['Output'])
            
            # Convert to long format for easier plotting
            gender_output_long = gender_output_counts.reset_index().melt(
                id_vars='Gender', 
                value_vars=[True, False] if False in gender_output_counts.columns else [True],
                var_name='Output', 
                value_name='Count'
            )
            
            # Create the bar chart
            gender_output_bar = px.bar(
                gender_output_long, 
                x='Gender', 
                y='Count', 
                color='Output',
                title='Gender vs Online Food Service Usage',
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            # Update legend labels
            gender_output_bar.update_layout(
                legend_title_text='Uses Online Food Service',
                legend=dict(
                    itemsizing='constant',
                    title_font_family='Arial',
                    font=dict(family='Arial', size=12),
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            
            # Update legend labels
            gender_output_bar.for_each_trace(lambda t: t.update(
                name='Yes' if t.name == 'True' else 'No'
            ))
            
        except Exception as e:
            print(f"Error creating gender_output_bar: {e}")
            gender_output_bar = px.bar(title="Gender vs Online Food Service Usage - Error")
            gender_output_bar.update_layout(
                annotations=[{
                    'text': 'Error creating chart. Insufficient data.',
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
        
        # Map scatter plot
        try:
            map_scatter = px.scatter_mapbox(
                filtered_df, 
                lat='latitude', 
                lon='longitude', 
                color='Output', 
                size='Age',
                title='Geographic Distribution of Online Food Service Usage',
                mapbox_style="open-street-map", 
                zoom=10,
                color_discrete_map={True: 'green', False: 'red'}
            )
            
            # Update legend labels
            map_scatter.for_each_trace(lambda t: t.update(
                name='Yes' if t.name == 'True' else 'No'
            ))
            
        except Exception as e:
            print(f"Error creating map_scatter: {e}")
            map_scatter = px.scatter(title="Geographic Distribution - Error")
            map_scatter.update_layout(
                annotations=[{
                    'text': 'Error creating map. Check latitude/longitude data.',
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
        
        # Pincode bar chart
        pincode_counts = filtered_df['Pin code'].value_counts().reset_index().head(10)
        pincode_counts.columns = ['Pin Code', 'Count']
        pincode_bar = px.bar(pincode_counts, 
                            x='Pin Code', y='Count', title='Top 10 Pin Codes by Count',
                            color='Pin Code', color_discrete_sequence=px.colors.qualitative.Pastel)
        
        # Age vs Family Size scatter plot
        age_family_scatter = px.scatter(filtered_df, x='Age', y='Family size', 
                                       color='Output', size='Age',
                                       title='Age vs Family Size by Online Food Service Usage',
                                       color_discrete_map={True: 'green', False: 'red'})
        
        # Update legend labels
        age_family_scatter.for_each_trace(lambda t: t.update(
            name='Yes' if t.name == 'True' else 'No'
        ))
        
        # Sunburst chart
        try:
            # Create a copy of the dataframe with string values for Output
            sunburst_df = filtered_df.copy()
            sunburst_df['Output'] = sunburst_df['Output'].map({True: 'Yes', False: 'No'})
            
            sunburst = px.sunburst(
                sunburst_df, 
                path=['Gender', 'Marital Status', 'Output'], 
                title='Hierarchical View of Online Food Service Usage',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
        except Exception as e:
            print(f"Error creating sunburst: {e}")
            sunburst = px.pie(title="Hierarchical View - Error")
            sunburst.update_layout(
                annotations=[{
                    'text': 'Error creating chart. Insufficient data for hierarchy.',
                    'showarrow': False,
                    'font': {'size': 20}
                }]
            )
        
        # Generate insights
        insights = []
        
        # Insight 1: Age
        avg_age = filtered_df['Age'].mean()
        insights.append(html.Li(f"Average age of the selected population: {avg_age:.1f} years"))
        
        # Insight 2: Gender
        gender_pct = filtered_df['Gender'].value_counts(normalize=True) * 100
        if not gender_pct.empty:
            top_gender = gender_pct.index[0]
            top_gender_pct = gender_pct.iloc[0]
            insights.append(html.Li(f"{top_gender}s make up {top_gender_pct:.1f}% of the selected population"))
        
        # Insight 3: Online Food Usage
        usage_count = filtered_df['Output'].sum()
        total_count = len(filtered_df)
        usage_pct = (usage_count / total_count) * 100 if total_count > 0 else 0
        insights.append(html.Li(f"{usage_pct:.1f}% of the selected population uses online food services ({usage_count} out of {total_count})"))
        
        # Insight 4: Age and Usage
        users = filtered_df[filtered_df['Output'] == True]
        non_users = filtered_df[filtered_df['Output'] == False]
        
        if not users.empty and not non_users.empty:
            avg_age_users = users['Age'].mean()
            avg_age_non_users = non_users['Age'].mean()
            age_diff = avg_age_users - avg_age_non_users
            if abs(age_diff) > 1:
                if age_diff > 0:
                    insights.append(html.Li(f"Users of online food services are on average {abs(age_diff):.1f} years older than non-users"))
                else:
                    insights.append(html.Li(f"Users of online food services are on average {abs(age_diff):.1f} years younger than non-users"))
        
        # Insight 5: Occupation and Usage
        try:
            # Create a crosstab with counts
            occupation_usage_counts = pd.crosstab(filtered_df['Occupation'], filtered_df['Output'])
            
            # Calculate percentages
            occupation_usage_pct = occupation_usage_counts.div(
                occupation_usage_counts.sum(axis=1), axis=0
            ) * 100
            
            if not occupation_usage_pct.empty and True in occupation_usage_pct.columns:
                top_occupation = occupation_usage_pct[True].idxmax()
                top_occupation_pct = occupation_usage_pct.loc[top_occupation, True]
                top_occupation_count = occupation_usage_counts.loc[top_occupation, True]
                total_occupation = occupation_usage_counts.loc[top_occupation].sum()
                
                insights.append(html.Li(
                    f"{top_occupation}s have the highest online food service usage at {top_occupation_pct:.1f}% "
                    f"({top_occupation_count} out of {total_occupation})"
                ))
        except Exception as e:
            print(f"Error generating occupation insight: {e}")
        
        # Insight 6: Feedback
        positive_count = (filtered_df['Feedback'] == 'Positive').sum()
        total_feedback = len(filtered_df)
        positive_pct = (positive_count / total_feedback) * 100 if total_feedback > 0 else 0
        
        insights.append(html.Li(
            f"{positive_pct:.1f}% of the feedback is positive "
            f"({positive_count} out of {total_feedback})"
        ))
        
        # Insight 7: Sample size
        insights.append(html.Li(
            f"Current selection includes {len(filtered_df)} individuals out of {len(df)} total "
            f"({len(filtered_df)/len(df)*100:.1f}%)"
        ))
        
        return (age_hist, gender_pie, marital_bar, occupation_bar, income_bar, education_bar, 
                family_bar, income_edu_heatmap, output_pie, feedback_pie, age_output_box, 
                gender_output_bar, map_scatter, pincode_bar, age_family_scatter, sunburst, insights)
    
    except Exception as e:
        print(f"Error in update_graphs: {e}")
        print(traceback.format_exc())
        
        # Return empty figures with error message
        error_fig = px.scatter(title="Error")
        error_fig.update_layout(
            annotations=[{
                'text': f'An error occurred: {str(e)}',
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
        
        return [error_fig] * 16 + [[html.Li(f"An error occurred: {str(e)}")]]

# Run the app
if __name__ == '__main__':
    try:
        print("Starting dashboard server...")
        app.run_server(debug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
        print(traceback.format_exc()) 