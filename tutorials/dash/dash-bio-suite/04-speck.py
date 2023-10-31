import urllib.request as urlreq
from dash import Dash, dcc, html, Input, Output, callback
import dash_bio as dashbio
from dash_bio.utils import xyz_reader
from flask import Flask
import os 

server = Flask(__name__)
app = Dash(server=server)

# To retrieve data from original source, uncomment 
# data = urlreq.urlopen(
#     'https://git.io/speck_methane.xyz'
# ).read().decode('utf-8')

dir_name = os.path.dirname(__file__)
data_path = dir_name + "/data/speck_methane.xyz.txt"
with open(data_path, 'r') as file:
    data = file.read()
    data = xyz_reader.read_xyz(datapath_or_datastring=data, is_datafile=False)

app.layout = html.Div([
    dcc.Dropdown(
        id='default-speck-preset-views',
        options=[
            {'label': 'Default', 'value': 'default'},
            {'label': 'Ball and stick', 'value': 'stickball'}
        ],
        value='default'
    ),
    dashbio.Speck(
        id='default-speck',
        data=data
    ),
])

@callback(
    Output('default-speck', 'presetView'),
    Input('default-speck-preset-views', 'value')
)
def update_preset_view(preset_name):
    return preset_name


if __name__ == '__main__':
    app.run(debug=True)
