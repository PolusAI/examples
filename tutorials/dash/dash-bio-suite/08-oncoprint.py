import json
import urllib.request as urlreq
from dash import Dash, html, Input, Output, callback
import dash_bio as dashbio
from flask import Flask

server = Flask(__name__)
app = Dash(server=server)



data = urlreq.urlopen(
    'https://git.io/oncoprint_dataset3.json'
).read().decode('utf-8')

data = json.loads(data)

app.layout = html.Div([
    dashbio.OncoPrint(
        id='dashbio-default-oncoprint',
        data=data
    ),
    html.Div(id='default-oncoprint-output')
])

@callback(
    Output('default-oncoprint-output', 'children'),
    Input('dashbio-default-oncoprint', 'eventDatum')
)
def update_output(event_data):
    if event_data is None or len(event_data) == 0:
        return 'There are no event data. Hover over or click on a part \
        of the graph to generate event data.'

    event_data = json.loads(event_data)

    return [
        html.Div('{}: {}'.format(
            key,
            str(event_data[key]).replace('<br>', '\n')
        ))
        for key in event_data.keys()]

if __name__ == '__main__':
    app.run(debug=True)
