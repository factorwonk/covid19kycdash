import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd


# PATH to csv file
PATH = 'client_sentiment.csv'
df = pd.read_csv(PATH).drop('Unnamed: 0', axis=1).reset_index()
# Transform the dataframe to get the mean daily VADER Score for each client
df = df.groupby(['Date','Client'])['VADER Score'].mean().reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df[df.Date > '2020-03-01']
fig = px.line(df, x='Date', y='VADER Score', color='Client')
fig.update_traces(mode='markers+lines')
#fig.show()

# Import CSS sheets from here
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H3("Market sentiment related to companies over time based on Covid-19 related news"),
    dcc.Graph(
        id = 'SentimentChart',
        figure=fig
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)