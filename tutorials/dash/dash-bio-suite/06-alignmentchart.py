import dash_bio as dashbio
from dash import Dash, html, Input, Output, callback
import urllib.request as urlreq
from flask import Flask

server = Flask(__name__)
app = Dash(server=server)



data = urlreq.urlopen(
    'https://git.io/alignment_viewer_p53.fasta'
).read().decode('utf-8')

app.layout = html.Div([
    dashbio.AlignmentChart(
        id='my-default-alignment-viewer',
        data=data,
        height=900,
        tilewidth=30,
    ),
    html.Div(id='default-alignment-viewer-output')
])

@callback(
    Output('default-alignment-viewer-output', 'children'),
    Input('my-default-alignment-viewer', 'eventDatum')
)
def update_output(value):
    if value is None:
        return 'No data.'
    return str(value)

if __name__ == '__main__':
    app.run(debug=True)
