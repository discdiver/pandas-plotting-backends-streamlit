import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair_pandas  # not just altair
import seaborn as sns
import altair

# import hvplot
# import hvplot.pandas
# import holoviews
import pandas_bokeh
import plotly
import plotly.express
import plotly.graph_objects as go
from bokeh.plotting import output_notebook


st.set_page_config(layout="wide")

plot_types = ("Scatter", "Histogram", "Bar", "Line")
backends = (
    "plotly",
    "altair",
    "pandas_bokeh",
    # "hvplot",    # glitchy
    # "holoviews", # glitchy
    "matplotlib",
)

# get data
@st.cache(allow_output_mutation=True)
def load_penguins():
    return sns.load_dataset("penguins")


pens_df = load_penguins()
df = pens_df.copy()
df.index = pd.date_range(start="1/1/18", periods=len(df), freq="D")

# title
with st.container():
    st.title("Explore Python pandas plotting backend options")

chart_type = st.selectbox("Choose your chart type", plot_types)


# Output all plots of selected type for the various pandas plotting backends
def df_plot(backend: str, chart_type: str, df: pd.DataFrame):
    """return pandas plots

    Args:
        backend: plotting library
        chart_type: type of plot
        df: penguins data
    """

    if backend == "matplotlib":
        fig, ax = plt.subplots()
        if chart_type == "Scatter":
            df["color"] = df["species"].replace(
                {"Adelie": "blue", "Chinstrap": "orange", "Gentoo": "green"}
            )

            f = df.plot(
                kind="scatter",
                x="bill_depth_mm",
                y="bill_length_mm",
                c="color",
                title="Bill Depth by Bill Length",
                ax=ax,
            )

        elif chart_type == "Histogram":

            f = df["bill_depth_mm"].plot(
                kind="hist",
                title="Count of Bill Depth Observations",
                ax=ax,
            )

        elif chart_type == "Bar":

            f = (
                df.groupby("species")
                .mean()
                .plot(
                    kind="bar",
                    y="bill_depth_mm",
                    title="Mean Bill Depth by Species",
                    ax=ax,
                )
            )

        elif chart_type == "Line":

            f = df.reset_index().plot(
                kind="line",
                x="index",
                y="bill_length_mm",
                title="Bill Length Over Time",
                ax=ax,
            )

    else:  # not matplotlib
        if chart_type == "Scatter":
            df["color"] = df["species"].replace(
                {"Adelie": "blue", "Chinstrap": "orange", "Gentoo": "green"}
            )
            if backend == "altair":
                fig = df.plot(
                    kind="scatter",
                    x="bill_depth_mm",
                    y="bill_length_mm",
                    c="color",
                    title="Bill Depth by Bill Length",
                )
            else:  # color is on  the way for bokeh with pandas, it appears
                fig = df.plot(
                    kind="scatter",
                    x="bill_depth_mm",
                    y="bill_length_mm",
                    color="color",
                    title="Bill Depth by Bill Length",
                )

        elif chart_type == "Histogram":

            fig = df["bill_depth_mm"].plot(
                kind="hist",
                title="Count of Bill Depth Observations",
            )

        elif chart_type == "Bar":

            fig = (
                df.groupby("species")
                .mean()
                .plot(
                    kind="bar",
                    y="bill_depth_mm",
                    title="Mean Bill Depth by Species",
                )
            )

        elif chart_type == "Line":

            fig = df.reset_index().plot(
                kind="line",
                x="index",
                y="bill_length_mm",
                title="Bill Length Over Time",
            )

    return fig


def show_plot(backend: str):
    """set plotting backends

    Args:
        backend: the plotting library backend name for pandas"""

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


for backend in backends:
    st.subheader(backend.title().split("_")[-1])
    pd.options.plotting.backend = backend
    output_notebook()  # required so you don't open new browser tabs every run
    show_plot(backend=backend)


with st.container():
    show_data = st.checkbox("See the raw data?")

    if show_data:
        df

    st.subheader("Notes")
    st.write(
        """
        - This app uses [Streamlit](https://streamlit.io/) and the [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) dataset.      
        - See the full code and check out the [GitHub repo](https://github.com/discdiver/pandas-plotting-backends-streamlit)
        - The pandas code is almost the same for each plot. 🐼 
        - Lineplots should be used with sequence data, so I created a date index with a sequence of dates for plotting. 
        - The pandas plotting API is not fully supported by these plotting backend libraries.

          
        Made by [Jeff Hale](https://jeffhale.net). 
        
        Subscribe to my [Data Awesome newsletter](https://dataawesome.com) for the latest tools, tips, and resources.
        """
    )
