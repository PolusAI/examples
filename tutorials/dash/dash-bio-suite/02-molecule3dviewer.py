import dash_bio as dashbio
from dash import Dash, html, Input, Output, callback
from dash_bio.utils import PdbParser, create_mol3d_style
from flask import Flask
import os 

server = Flask(__name__)
app = Dash(server=server)

# To retrieve data from original source, uncomment 
# parser = PdbParser('https://git.io/4K8X.pdb')

dir_name = os.path.dirname(__file__)
parser = PdbParser(dir_name + "/data/4K8X.pdb.txt")
data = parser.mol3d_data()
styles = create_mol3d_style(
    data['atoms'], visualization_type='cartoon', color_element='residue'
)

app.layout = html.Div([
    dashbio.Molecule3dViewer(
        id='dashbio-default-molecule3d',
        modelData=data,
        styles=styles
    ),
    "Selection data",
    html.Hr(),
    html.Div(id='default-molecule3d-output')
])

@callback(
    Output('default-molecule3d-output', 'children'),
    Input('dashbio-default-molecule3d', 'selectedAtomIds')
)
def show_selected_atoms(atom_ids):
    if atom_ids is None or len(atom_ids) == 0:
        return 'No atom has been selected. Click somewhere on the molecular \
        structure to select an atom.'
    return [html.Div([
        html.Div('Element: {}'.format(data['atoms'][atm]['elem'])),
        html.Div('Chain: {}'.format(data['atoms'][atm]['chain'])),
        html.Div('Residue name: {}'.format(data['atoms'][atm]['residue_name'])),
        html.Br()
    ]) for atm in atom_ids]

if __name__ == '__main__':
    app.run(debug=True)
