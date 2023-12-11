import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback
import dash_bio as dashbio
from flask import Flask
import os 

server = Flask(__name__)
app = Dash(server=server)

# To retrieve data from original source, uncomment 
# df = pd.read_csv('https://git.io/volcano_data1.csv')

dir_name = os.path.dirname(__file__)
df = pd.read_csv(dir_name + '/data/volcano_data1.csv')


app.layout = html.Div([
    'Effect sizes',
    dcc.RangeSlider(
        id='default-volcanoplot-input',
        min=-3,
        max=3,
        step=0.05,
        marks={i: {'label': str(i)} for i in range(-3, 3)},
        value=[-0.5, 1]
    ),
    html.Br(),
    html.Div(
        dcc.Graph(
            id='dashbio-default-volcanoplot',
            figure=dashbio.VolcanoPlot(
                dataframe=df
            )
        )
    )
])

@callback(
    Output('dashbio-default-volcanoplot', 'figure'),
    Input('default-volcanoplot-input', 'value')
)
def update_volcanoplot(effects):
    return dashbio.VolcanoPlot(
        dataframe=df,
        genomewideline_value=2.5,
        effect_size_line=effects
    )

if __name__ == '__main__':
    app.run(debug=True)
