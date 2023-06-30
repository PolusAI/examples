import os
import pandas as pd
from flask import Flask
from dash.dependencies import Input, Output
from dash import Dash, html, dcc
import dash_bio as dashbio
from dash import dash_table
from dash.dash_table.Format import Group
from Bio import Entrez
from xml.etree import ElementTree as ET
import re


def verify_email(user_email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if user_email and re.match(pattern, user_email):
        entrez_email = user_email
    else:
        entrez_email = "user@nih.gov"
    return entrez_email


def parse(string):
    parts = string.split("<br>")
    snp = parts[0].split(": ")[1]
    gene = parts[1].split(": ")[1]
    return snp, gene


def get_ncbi_sum(gene_name):
    user_email = os.environ.get("JUPYTERHUB_USER")

    entrez_email = verify_email(user_email)

    Entrez.email = entrez_email
    handle = Entrez.esearch(
        db="gene", term=f'{gene_name}[Gene Name] AND "Homo sapiens"[Organism]', retmax=1
    )
    record = Entrez.read(handle)
    if not record["IdList"]:
        gene_summary = (
            "The NCBI Gene database does not have a summary description of this gene."
        )
    else:
        gene_id = record["IdList"][0]
        handle = Entrez.esummary(db="gene", id=gene_id)
        record = Entrez.read(handle)
        gene_summary = record["DocumentSummarySet"]["DocumentSummary"][0]["Summary"]
        gene_summary = str(gene_summary)
        if len(gene_summary) == 0:
            gene_summary = "The NCBI Gene database does not have a summary description of this gene."

    return gene_summary


def get_table(data):
    columns = [
        {"name": "Significant Snp", "id": "snp"},
        {"name": "-log10(P)", "id": "p"},
        {"name": "Associated Gene", "id": "gene"},
        {"name": "Gene Summary", "id": "gene summary"},
    ]

    max_height = 50

    table_rows = []
    for row in data:
        truncated_description = (
            row["gene summary"][:max_height] + "..."
            if len(row["gene summary"]) > max_height
            else row["gene summary"]
        )
        table_rows.append(
            {
                "snp": row["snp"],
                "p": row["p"],
                "gene": row["gene"],
                "gene summary": truncated_description,
            }
        )

    table = dash_table.DataTable(
        columns=columns,
        data=table_rows,
        style_cell={
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "maxHeight": f"{max_height}px",
            "minWidth": "100px",
            "maxWidth": "300px",
            "whiteSpace": "nowrap",
            "textAlign": "left",
            "padding": "5px",
        },
        tooltip_header={
            "id": "Full Text",
            "name": "Full Text",
            "description": "Full Text",
            "backgroundColor": "rgb(204, 230, 255)",
            "fontWeight": "bold",
        },
        tooltip_data=[
            {
                column["id"]: {"value": str(data_row[column["id"]]), "type": "markdown"}
                for column in columns
            }
            for data_row in data
        ],
    )

    return table


server = Flask(__name__)
app = Dash(server=server)

df = pd.read_csv("manhattan_data.csv")


app.layout = html.Div(
    [
        html.H1(
            "Dash Bio Manhattan Plot",
            style={"text-align": "center", "color": "blue", "font-weight": "bold"},
        ),
        html.Div(
            dcc.Graph(
                id="default-dashbio-manhattanplot",
                figure=dashbio.ManhattanPlot(dataframe=df),
            )
        ),
        html.Br(),
        "Threshold value",
        dcc.Slider(
            id="default-manhattanplot-input",
            min=1,
            max=10,
            marks={i: {"label": str(i)} for i in range(10)},
            value=6,
        ),
        html.Br(),
        html.P(
            "Note: The manhattan plot dataset that is provided by Dash Plotly appears to be a synthetic example that was generated for the purpose of demonstrating the functionality of the Manhattan plot in Dash Bio. "
            "It is not intended to represent real-world data or any specific trait or disease nor does the Dash Bio "
            "documentation provide any information relating to this.",
            style={
                "text-align": "center",
                "border": "2px solid green",
                "padding": "10px",
            },
        ),
        html.Br(),
        html.Div(id="table-container"),
    ]
)


@app.callback(
    Output("default-dashbio-manhattanplot", "figure"),
    Input("default-manhattanplot-input", "value"),
)
def update_manhattanplot(threshold):
    return dashbio.ManhattanPlot(dataframe=df, genomewideline_value=threshold)


# Add table
@app.callback(
    Output("table-container", "children"),
    Input("default-dashbio-manhattanplot", "figure"),
)
def get_significant_snps(fig):
    table_rows = []
    data = fig["data"][0]
    top_snps = list(zip(data["text"], data["y"]))
    for top_snp in top_snps:
        p = top_snp[1]
        snp, gene = parse(top_snp[0])
        gene_sum = get_ncbi_sum(gene)
        table_rows.append({"snp": snp, "p": p, "gene": gene, "gene summary": gene_sum})
        table_rows = sorted(table_rows, key=lambda x: x["p"], reverse=True)

    table = get_table(table_rows)

    return table


if __name__ == "__main__":
    app.run_server(debug=True)
