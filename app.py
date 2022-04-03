import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair_pandas  # not just altair
import seaborn as sns
import altair

import hvplot
import hvplot.pandas
import holoviews
import pandas_bokeh
import plotly
import plotly.express
import plotly.graph_objects as go
from bokeh.plotting import output_notebook


# can only set this once, first thing to set
st.set_page_config(layout="wide")

plot_types = ("Scatter", "Histogram", "Bar", "Line")
backends = (
    "plotly",
    "altair",
    "pandas_bokeh",
    # "hvplot",
    #  "holoviews",
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
    st.write(
        """
    - Most plots are interactive and downloadable.
    - The pandas code is almost the same for each plot. 🐼 
    - See and fork the code on [GitHub](https://github.com/discdiver/pandas-plotting-backends-streamlit)
    
    """
    )

chart_type = st.selectbox("Choose your chart type", plot_types)

# layout
with st.container():
    st.subheader(f"Showing:  {chart_type}")
    st.write("")


# two_cols = st.checkbox("2 columns?", True)
# if two_cols:
#     col1, col2 = st.columns(2)


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
    # if two_cols:
    #     with col1:
    #         st.write("Plotly")
    #         pd.options.plotting.backend = "plotly"
    #         show_plot(backend="plotly")
    #     with col2:
    #         st.write("Altair")
    #         pd.options.plotting.backend = "altair"
    #         show_plot(backend="altair")
    #     # with col2:
    #     #     show_plot(backend="holoviews")
    #     # with col1:
    #     #     show_plot(backend="hvplot")
    #     # # with col1:
    #         # st.write("Bokeh (pandas_bokeh)")
    #         # pd.options.plotting.backend = "pandas_bokeh"
    #         # output_notebook()  # required so you don't open new browser tabs every run
    #         # show_plot(backend="pandas_bokeh")
    #     with col2:
    #         st.write("Matplotlib")
    #         pd.options.plotting.backend = "matplotlib"
    #         show_plot(backend="matplotlib")
    #     with col1:
    #         st.write("Bokeh")
    #         pd.options.plotting.backend = "pandas_bokeh"
    #         show_plot(backend="pandas_bokeh")

    # else:
    with st.container():
        for backend in backends:
            st.write(backend.title().split("_")[-1])
            pd.options.plotting.backend = backend
            output_notebook()  # required so you don't open new browser tabs ever run
            show_plot(backend=backend)


# display data
with st.container():
    show_data = st.checkbox("See the raw data?")

    if show_data:
        df

    # notes
    st.subheader("Notes")
    st.write(
        """
        - This app uses [Streamlit](https://streamlit.io/) and the [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) dataset.      
        - To see the full code check out the [GitHub repo](https://github.com/discdiver/pandas-plotting-backends-streamlit)
        - Lineplots should be used with sequence data, so I created a date index with a sequence of dates for plotting. 
        - The pandas plotting API is not fully supported by these plotting backend libraries, which is unfortunate because learning one plotting API would be nicer than learning a half-dozen. 😀
        - Check out my example app with a half-dozen Python plotting libraries and code [here](https://share.streamlit.io/discdiver/pandas-plotting-backends-streamlit/main/app.py).
        
        Made by [Jeff Hale](https://www.linkedin.com/in/-jeffhale/). 
        
        Subscribe to my [Data Awesome newsletter](https://dataawesome.com) for the latest tools, tips, and resources.
        """
    )
