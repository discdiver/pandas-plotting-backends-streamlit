import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
from bokeh.plotting import figure

# get data
@st.cache(allow_output_mutation=True)
def load_penguins():
    return sns.load_dataset("penguins")


pens_df = load_penguins()
df = pens_df.copy()
df.index = pd.date_range(start="1/1/18", periods=len(df), freq="D")


with st.beta_container():
    st.title("Python Data Visualization Tour")
    st.header("Popular plots in popular plotting libraries")
    st.write("""See the code and plots for five libraries at once.""")


# User choose user type
chart_type = st.selectbox("Choose your chart type", plot_types)

with st.beta_container():
    st.subheader(f"Showing:  {chart_type}")
    st.write("")

three_cols = st.checkbox("2 columns?")
if three_cols:
    col1, col2 = st.beta_columns(2)


"Output all the plots from the pandas plotting backend "