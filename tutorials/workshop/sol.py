# import dependencies
import solara
import plotly.express as px
import pandas as pd
from typing import Optional, cast
from solara.components.file_drop import FileDrop

# load sample dataset
df_sample = px.data.iris()

# state management
class State:
    # set reactive variables to use with widgets
    color = solara.reactive(cast(Optional[str], None))
    x = solara.reactive(cast(Optional[str], None))
    y = solara.reactive(cast(Optional[str], None))
    df = solara.reactive(cast(Optional[pd.DataFrame], None))

    # assign initial values when loading sample dataset
    @staticmethod
    def load_sample():
        State.x.value = str("sepal_length")
        State.y.value = str("sepal_width")
        State.color.value = str("species")
        State.df.value = df_sample
    
    # assign initial values when uploading custom dataset
    @staticmethod
    def load_from_file(file):
        df = pd.read_csv(file["file_obj"])
        State.x.value = str(df.column[0])
        State.y.value = str(df.column[1])
        State.color.value = str(df.column[2])
        State.df.value = df
    
    # clear stored dataframe upon reset
    @staticmethod
    def reset():
        State.df.value = None

# define app's main component
@solara.component
def Page():
    # initialize variable from state
    df = State.df.value

    # name app
    with solara.AppBarTitle():
        solara.Text("Solara Demo")
    
    # create sidebar panel
    with solara.Sidebar():
        with solara.Column():
            # put buttons in a row
            with solara.Row():
                solara.Button("Sample dataset", color="primary", text=True, outlined=True, on_click=State.load_sample, disabled=df is not None)
                solara.Button("Clear dataset", color="primary", text=True, outlined=True, on_click=State.reset)
            FileDrop(on_file=State.load_from_file, on_total_progress=lambda *args: None, label="Drag file here")
        # put selection widgets in a column inside sidebar panel
        if df is not None:
            columns = list(map(str, df.columns))
            solara.Select("X-axis", values=columns, value=State.x)
            solara.Select("Y-axis", values=columns, value=State.y)
            solara.Select("Color", values=columns, value=State.color)
    
    # create scatter plot
    if df is not None:
        if State.x.value and State.y.value:
            fig = px.scatter(
                df,
                State.x.value,
                State.y.value,
                color = State.color.value
            )
            solara.FigurePlotly(fig)
            solara.DataFrame(df, items_per_page=10)
        else:
            solara.Warning("Select x and y axes")
    else:
        solara.Info("No data loaded")

# app layout
@solara.component
def Layout(children):
    route, routes = solara.use_route()
    return solara.AppLayout(children=children)