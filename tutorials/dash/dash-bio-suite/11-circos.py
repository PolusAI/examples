import json
import urllib.request as urlreq
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bio as dashbio
from flask import Flask
import os 

server = Flask(__name__)
app = Dash(server=server)

# To retrieve data from original source, uncomment 
# data = urlreq.urlopen(
#     "https://git.io/circos_graph_data.json"
# ).read().decode("utf-8")
# circos_graph_data = json.loads(data)

dir_name = os.path.dirname(__file__)
data_path = dir_name + "/data/circos_graph_data.json"
with open(data_path, 'r') as data:
    circos_graph_data = json.load(data)


app.layout = html.Div(
    [
        dashbio.Circos(
            id="my-dashbio-default-circos",
            layout=circos_graph_data["GRCh37"],
            selectEvent={"0": "hover"},
            tracks=[
                {
                    "type": "CHORDS",
                    "data": circos_graph_data["chords"],
                    "config": {
                        "tooltipContent": {
                            "source": "source",
                            "sourceID": "id",
                            "target": "target",
                            "targetID": "id",
                            "targetEnd": "end",
                        }
                    },
                }
            ],
        ),
        "Graph type:",
        dcc.Dropdown(
            id="histogram-chords-default-circos",
            options=[{"label": x, "value": x} for x in ["histogram", "chords"]],
            value="chords",
        ),
        "Event data:",
        html.Div(id="default-circos-output"),
    ]
)


@callback(
    Output("default-circos-output", "children"),
    Input("my-dashbio-default-circos", "eventDatum"),
)
def update_output(value):
    if value is not None:
        return [html.Div("{}: {}".format(v.title(), value[v])) for v in value.keys()]
    return "There are no event data. Hover over a data point to get more information."


@callback(
    Output("my-dashbio-default-circos", "tracks"),
    Input("histogram-chords-default-circos", "value"),
    State("my-dashbio-default-circos", "tracks"),
)
def change_graph_type(value, current):
    if value == "histogram":
        current[0].update(data=circos_graph_data["histogram"], type="HISTOGRAM")

    elif value == "chords":
        current[0].update(
            data=circos_graph_data["chords"],
            type="CHORDS",
            config={
                "tooltipContent": {
                    "source": "source",
                    "sourceID": "id",
                    "target": "target",
                    "targetID": "id",
                    "targetEnd": "end",
                }
            },
        )
    return current


if __name__ == "__main__":
    app.run(debug=True)
