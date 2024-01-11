# import dependencies
import streamlit as st
import pandas as pd
import plotly.express as px

# set up tabs
tab1, tab2 = st.tabs(["Sample: Iris", "Upload Data"])

# tab for sample dataset
with tab1:
    # load data
    iris = px.data.iris()

    # editable dataframe
    with st.expander("See & Edit Data"):
        edited_df = st.data_editor(
            iris, # source dataframe
            num_rows="dynamic", # enable addition and removal of rows
            hide_index=True
        )

    # add widgets for x-axis, y-axis, and color selection
    x_ax = st.selectbox(
        "Select X-axis",
        iris.columns,
        index=0
    )

    y_ax = st.selectbox(
        "Select Y-axis",
        iris.columns,
        index=1
    )

    color = st.selectbox(
        "Select Color",
        iris.columns,
        index=4
    )

    # plotly figure
    fig = px.scatter(
            edited_df,
            x = x_ax,
            y = y_ax,
            color = color)
    st.plotly_chart(fig, use_container_width=True)

# tab for uploaded file
with tab2:
    # upload data
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # editable dataframe
        with st.expander("See & Edit Data"):
            edited_data = st.data_editor(
                data, # source dataframe
                num_rows="dynamic", # enable addition and removal of rows
                hide_index=True
            )

        # add widgets for x- axis, y-axis, and color selection
        x_ax2 = st.selectbox(
            "Select X-axis",
            edited_data.columns,
            index=0
        )

        y_ax2 = st.selectbox(
            "Select Y-axis",
            edited_data.columns,
            index=1
        )

        color2 = st.selectbox(
            "Select Color",
            edited_data.columns,
            index=None
        )

        # plotly figure
        fig2 = px.scatter(
                edited_data,
                x = x_ax2,
                y = y_ax2,
                color = color2)
        st.plotly_chart(fig2, use_container_width=True)
