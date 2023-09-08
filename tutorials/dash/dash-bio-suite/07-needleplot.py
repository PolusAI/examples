import json
import urllib.request as urlreq
from dash import Dash, html, dcc, Input, Output, callback
import dash_bio as dashbio
from flask import Flask

server = Flask(__name__)
app = Dash(server=server)



data = urlreq.urlopen(
    'https://git.io/needle_PIK3CA.json'
).read().decode('utf-8')

mdata = json.loads(data)

app.layout = html.Div([
    "Show or hide range slider",
    dcc.Dropdown(
        id='default-needleplot-rangeslider',
        options=[
            {'label': 'Show', 'value': 1},
            {'label': 'Hide', 'value': 0}
        ],
        clearable=False,
        multi=False,
        value=1,
        style={'width': '400px'}
    ),
    dashbio.NeedlePlot(
        id='dashbio-default-needleplot',
        mutationData=mdata
    )
])

@callback(
    Output('dashbio-default-needleplot', 'rangeSlider'),
    Input('default-needleplot-rangeslider', 'value')
)
def update_needleplot(show_rangeslider):
    return True if show_rangeslider else False

if __name__ == '__main__':
    app.run(debug=True)
