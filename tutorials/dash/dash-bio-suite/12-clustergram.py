import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback
import dash_bio as dashbio
from flask import Flask

server = Flask(__name__)
app = Dash(server=server)

# To retrieve data from original source, uncomment 
# df = pd.read_csv('https://git.io/clustergram_brain_cancer.csv').set_index('ID_REF')

df = pd.read_csv('/Users/aditi.patel/Desktop/GitHub/examples/tutorials/dash/dash-bio-suite/data/clustergram_brain_cancer.csv').set_index('ID_REF')

columns = list(df.columns.values)
rows = list(df.index)

app.layout = html.Div([
    "Rows to display",
    dcc.Dropdown(
        id='my-default-clustergram-input',
        options=[
            {'label': row, 'value': row} for row in list(df.index)
        ],
        value=rows[:10],
        multi=True
    ),

    html.Div(id='my-default-clustergram')
])

@callback(
    Output('my-default-clustergram', 'children'),
    Input('my-default-clustergram-input', 'value')
)
def update_clustergram(rows):
    if len(rows) < 2:
        return "Please select at least two rows to display."

    return dcc.Graph(figure=dashbio.Clustergram(
        data=df.loc[rows].values,
        column_labels=columns,
        row_labels=rows,
        color_threshold={
            'row': 250,
            'col': 700
        },
        hidden_labels='row',
        height=800,
        width=700
    ))

if __name__ == '__main__':
    app.run(debug=True)
