from dash import Dash, html
import dash_bio as dashbio
from flask import Flask

server = Flask(__name__)
app = Dash(server=server)


app.layout = html.Div([
    dashbio.Jsme(),
])

if __name__ == '__main__':
    app.run(debug=True)
