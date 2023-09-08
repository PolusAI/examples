from dash import Dash, html, dcc, Input, Output, callback
import dash_bio as dashbio
from dash.exceptions import PreventUpdate
from flask import Flask

server = Flask(__name__)
app = Dash(server=server)


sequences = {
    'PDB_01019': {
        'sequence': 'AUGGGCCCGGGCCCAAUGGGCCCGGGCCCA',
        'structure': '.((((((())))))).((((((()))))))'
    },
    'PDB_00598': {
        'sequence': 'GGAGAUGACgucATCTcc',
        'structure': '((((((((()))))))))'
    }
}

app.layout = html.Div([
    dashbio.FornaContainer(id='my-default-forna'),
    html.Hr(),
    html.P('Select the sequences to display below.'),
    dcc.Dropdown(
        id='my-default-forna-sequence-display',
        options=[
            {'label': name, 'value': name} for name in sequences.keys()
        ],
        multi=True,
        value=['PDB_01019']
    )
])

@callback(
    Output('my-default-forna', 'sequences'),
    Input('my-default-forna-sequence-display', 'value')
)
def show_selected_sequences(value):
    if value is None:
        raise PreventUpdate
    return [
        sequences[selected_sequence]
        for selected_sequence in value
    ]

if __name__ == '__main__':
    app.run(debug=True)
