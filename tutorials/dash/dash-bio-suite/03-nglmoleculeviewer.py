import dash_bio as dashbio
from dash import Dash, dcc, html, Input, Output, callback
from dash.exceptions import PreventUpdate
import dash_bio.utils.ngl_parser as ngl_parser
from flask import Flask

server = Flask(__name__)
app = Dash(server=server)


data_path = "https://raw.githubusercontent.com/plotly/datasets/master/Dash_Bio/Molecular/"


dropdown_options = [
    {"label": "1BNA", "value": "1BNA"},
    {"label": "MPRO", "value": "MPRO"},
    {"label": "PLPR", "value": "PLPR"},
    {"label": "5L73", "value": "5L73"},
    {"label": "NSP2", "value": "NSP2"}
]

app.layout = html.Div([
    dcc.Markdown('''
    ### NglMoleculeViewer Controls

    * Rotate Stage: Left-click on the viewer and move the mouse to rotate the stage.
    * Zoom: Use the mouse scroll-wheel to zoom in and out of the viewer.
    * Pan: Right click on the viewer to pan the stage.
    * Individual Molecule Interaction: Left click on the molecule to interact with, then hold the
    `CTRL` key and use right and left click mouse buttons to rotate and pan individual molecules.
    '''),
    dcc.Dropdown(
        id="default-ngl-molecule-dropdown",
        options=dropdown_options,
        placeholder="Select a molecule",
        value="1BNA"
    ),
    dashbio.NglMoleculeViewer(id="default-ngl-molecule"),
])

@callback(
    Output("default-ngl-molecule", 'data'),
    Output("default-ngl-molecule", "molStyles"),
    Input("default-ngl-molecule-dropdown", "value")
)
def return_molecule(value):

    if (value is None):
        raise PreventUpdate

    molstyles_dict = {
        "representations": ["cartoon", "axes+box"],
        "chosenAtomsColor": "white",
        "chosenAtomsRadius": 1,
        "molSpacingXaxis": 100,
    }

    data_list = [ngl_parser.get_data(data_path=data_path, pdb_id=value, color='red',reset_view=True, local=False)]

    return data_list, molstyles_dict

if __name__ == '__main__':
    app.run(debug=True)
