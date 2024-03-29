import urllib.request as urlreq
from dash import Dash, dcc, html, Input, Output, callback
import dash_bio as dashbio
from dash_bio.utils import protein_reader
from flask import Flask
import os 

server = Flask(__name__)
app = Dash(server=server)


# To retrieve data from original source, uncomment 
# fasta_str = urlreq.urlopen(
#     'https://git.io/sequence_viewer_P01308.fasta'
# ).read().decode('utf-8')

dir_name = os.path.dirname(__file__)
data_path = dir_name + "/data/sequence_viewer_P01308.fasta.txt"
with open(data_path, 'r') as file:
    fasta_str = file.read()

seq = protein_reader.read_fasta(datapath_or_datastring=fasta_str, is_datafile=False)[0]['sequence']

app.layout = html.Div([
    dashbio.SequenceViewer(
        id='default-sequence-viewer',
        sequence=seq
    ),
    html.Div(id='default-sequence-viewer-output')
])


@callback(
    Output('default-sequence-viewer-output', 'children'),
    Input('default-sequence-viewer', 'mouseSelection')
)
def update_output(value):
    if value is None or len(value) == 0:
        return 'There is no mouse selection.'
    return 'The mouse selection is {}.'.format(value['selection'])


if __name__ == '__main__':
    app.run(debug=True)
