import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np

# PATH to csv file
PATH = 'client_sentiment.csv'
df = pd.read_csv(PATH).drop('Unnamed: 0', axis=1).reset_index()

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

# Import CSS sheets from here
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H4(children='Covid-19 related news for customers'),
    generate_table(df)
])

if __name__ == '__main__':
    app.run_server(debug=True)