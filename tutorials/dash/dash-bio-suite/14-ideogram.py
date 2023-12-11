from dash import Dash, html, dcc, Input, Output, callback
import dash_bio as dashbio
from flask import Flask

server = Flask(__name__)
app = Dash(server=server)


app.layout = html.Div([
    'Select which chromosomes to display on the ideogram below:',
    dcc.Dropdown(
        id='my-default-displayed-chromosomes',
        options=[{'label': str(i), 'value': str(i)} for i in range(1, 23)],
        multi=True,
        value=[str(i) for i in range(1, 23)]
    ),
    dashbio.Ideogram(
        id='my-default-dashbio-ideogram'
    ),
    html.Div(id='my-default-ideogram-rotated')
])

@callback(
    Output('my-default-dashbio-ideogram', 'chromosomes'),
    Input('my-default-displayed-chromosomes', 'value')
)
def update_ideogram(value):
    return value

@callback(
    Output('my-default-ideogram-rotated', 'children'),
    Input('my-default-dashbio-ideogram', 'rotated')
)
def update_ideogram_rotated(rot):
    return 'You have {} selected a chromosome.'.format(
        '' if rot else 'not')

if __name__ == '__main__':
    app.run(debug=True)
