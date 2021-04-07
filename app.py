import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair_pandas  # not just altair
import seaborn as sns

# import hvplot         dependency  conflict with bokeh breaks if 2.2.3 so have to use 2.2.2
# import hvplot.pandas
# import holoviews
import pandas_bokeh
import plotly
import plotly.express
import plotly.graph_objects as go
from bokeh.plotting import output_notebook


# can only set this once, first thing to set
st.set_page_config(layout="wide")

plot_types = ("Scatter", "Histogram", "Bar", "Line")
backends = ("plotly", "altair", "pandas_bokeh")  # hvplot "holoviews", "matplotlib",

# get data
@st.cache(allow_output_mutation=True)
def load_penguins():
    return sns.load_dataset("penguins")


pens_df = load_penguins()
df = pens_df.copy()
df.index = pd.date_range(start="1/1/18", periods=len(df), freq="D")


with st.beta_container():
    st.title("Interactive Python pandas Plotting Backend Options")
    st.write("""Basically same pandas code. 🐼""")


# User choose user type
chart_type = st.selectbox("Choose your chart type", plot_types)

with st.beta_container():
    st.subheader(f"Showing:  {chart_type}")
    st.write("")

two_cols = st.checkbox("2 columns?")
if two_cols:
    col1, col2 = st.beta_columns(2)


# Output all the plots of the selected type for the various pandas plotting backends
def df_plot(backend: str, chart_type: str, df):
    """ return pandas plots """

    if chart_type == "Scatter":
        df["color"] = df["species"].replace(
            {"Adelie": "blue", "Chinstrap": "orange", "Gentoo": "green"}
        )
        if backend == 'altair':
            figgy = df.plot(
                kind="scatter",
                x="bill_depth_mm",
                y="bill_length_mm",
                c="color",  
                title="Bill Depth by Bill Length",
            )
        else:      # color is on  the way for bokeh with pandas, it looks like
            figgy = df.plot(
                kind="scatter",
                x="bill_depth_mm",
                y="bill_length_mm",
                color="color",  
                title="Bill Depth by Bill Length",
            )

    elif chart_type == "Histogram":
        
        figgy = df["bill_depth_mm"].plot(
            kind="hist",
            title="Count of Bill Depth Observations",
        )

    elif chart_type == "Bar":

        figgy = (
            df.groupby("species")
            .mean()
            .plot(
                kind="bar",
                y="bill_depth_mm",
                title="Mean Bill Depth by Species",
            )
        )
    # no boxplot for some options yet and issues
    # elif chart_type == "Boxplot":      
    #     figgy = df.plot(kind='box', y='species')

    elif chart_type == "Line":
        
        figgy = df.reset_index().plot(
            kind="line",
            x="index",
            y="bill_length_mm",
            title="Bill Length Over Time",
        )

    return figgy


# create plots
def show_plot(backend: str):

    plot = df_plot(backend, chart_type, df)

    if backend == "matplotlib":
        st.pyplot(plot)
    elif backend == "plotly":
        st.plotly_chart(plot, use_container_width=True)
    elif backend == "altair":
        st.altair_chart(plot, use_container_width=True)
    elif backend == "holoviews":
        st.bokeh_chart(plot, use_container_width=True)
    elif backend == "hvplot":
        st.bokeh_chart(plot, use_container_width=True)
    elif backend == "pandas_bokeh":
        st.bokeh_chart(plot, use_container_width=True)


# output plots
if two_cols:
    # with col1:
    #    show_plot(backend="matplotlib")
    with col1:
        pd.options.plotting.backend = "plotly"
        show_plot(backend="plotly")
    with col2:
        pd.options.plotting.backend = "altair"
        show_plot(backend="altair")
    # with col2:
    #     show_plot(backend="holoviews")
    # with col1:
    #     show_plot(backend="hvplot")
    with col1:
        pd.options.plotting.backend = "pandas_bokeh"
        output_notebook()         # required so you don't open new browser tabs ever run
        show_plot(backend="pandas_bokeh")
else:
    with st.beta_container():
        for backend in backends:
            pd.options.plotting.backend = backend
            output_notebook()         # required so you don't open new browser tabs ever run
            show_plot(backend=backend)

# display data
with st.beta_container():
    show_data = st.checkbox("See the raw data?")

    if show_data:
        df

    # notes
    st.subheader("Notes")
    st.write(
        """
        - This app uses [Streamlit](https://streamlit.io/) and the [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) dataset.      
        - To see the full code check out the [GitHub repo]
        - Lineplots should have sequence data, so I created a date index with a sequence of dates for them. 
        - You can choose to see two columns, but with a narrow screen this will switch to one column automatically.
        - The pandas plotting API is not fully supported by these plotting backend libraries, 
        which is unfortunate because learning one plotting API would be nicer than learning a half-dozen. 😀 
        
        Made by Jeff Hale. 
        
        Subscribe to my [Data Awesome newsletter](https://dataawesome.com) for the latest tools, tips, and resources.
        """
    )
