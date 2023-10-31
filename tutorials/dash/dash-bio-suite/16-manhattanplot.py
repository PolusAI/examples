import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback
import dash_bio as dashbio
from flask import Flask
import os 

server = Flask(__name__)
app = Dash(server=server)

# To retrieve data from original source, uncomment 
# df = pd.read_csv('https://git.io/manhattan_data.csv')

dir_name = os.path.dirname(__file__)
df = pd.read_csv(dir_name + '/data/manhattan_data.csv')

app.layout = html.Div([
    'Threshold value',
    dcc.Slider(
        id='default-manhattanplot-input',
        min=1,
        max=10,
        marks={
            i: {'label': str(i)} for i in range(10)
        },
        value=6
    ),
    html.Br(),
    html.Div(
        dcc.Graph(
            id='default-dashbio-manhattanplot',
            figure=dashbio.ManhattanPlot(
                dataframe=df
            )
        )
    )
])

@callback(
    Output('default-dashbio-manhattanplot', 'figure'),
    Input('default-manhattanplot-input', 'value')
)
def update_manhattanplot(threshold):

    return dashbio.ManhattanPlot(
        dataframe=df,
        genomewideline_value=threshold
    )

if __name__ == '__main__':
    app.run(debug=True)
