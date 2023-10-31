import json
import urllib.request as urlreq
from dash import Dash, html, Input, Output, callback
import dash_bio as dashbio
from flask import Flask
import os 

server = Flask(__name__)
app = Dash(server=server)

# To retrieve data from original source, uncomment 
# model_data = urlreq.urlopen(
#     'https://git.io/mol2d_buckminsterfullerene.json'
# ).read().decode('utf-8')
# model_data = json.load(model_data)
dir_name = os.path.dirname(__file__)
model_data_path = dir_name + '/data/mol2d_buckminsterfullerene.json'

with open(model_data_path, 'r') as model_data:
    model_data = json.load(model_data)

app.layout = html.Div([
    dashbio.Molecule2dViewer(
        id='dashbio-default-molecule2d',
        modelData=model_data
    ),
    html.Hr(),
    html.Div(id='default-molecule2d-output')
])

@callback(
    Output('default-molecule2d-output', 'children'),
    Input('dashbio-default-molecule2d', 'selectedAtomIds')
)
def update_selected_atoms(ids):
    if ids is None or len(ids) == 0:
        return "No atom has been selected. Select atoms by clicking on them."
    return "Selected atom IDs: {}.".format(', '.join([str(i) for i in ids]))

if __name__ == '__main__':
    app.run(debug=True)
