#!/usr/bin/env python3

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

df = pd.read_csv("BostonHousing.csv")

available_cols = df.columns
available_cols.sort_values()

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = (
    "Boston Housing Data"  # The title that will appear in the internet browser tab
)

colors = {"background": "#f0f0f0"}

app.layout = html.Div(
    [
        html.H2(children="Boston Housing Data", style={"textAlign": "left"}),
        html.Div(
            [
                html.H4("Filter", style={"textAlign": "left"}),
                html.H6("X-Axis", style={"textAlign": "left"}),
                dcc.Dropdown(
                    id="xaxis-dropdown",
                    options=[{"label": i, "value": i} for i in available_cols],
                    value="medv",
                ),
                html.H6("Y-Axis", style={"textAlign": "left"}),
                dcc.Dropdown(
                    id="yaxis-dropdown",
                    options=[{"label": i, "value": i} for i in available_cols],
                    value="rm",
                ),
                dcc.Markdown(
                    """
            **CRIM:** Per capita crime rate by town

            **ZN:** Proportion of residential land zoned for lots over 25,000 sq. ft

            **INDUS:** Proportion of non-retail business acres per town

            **CHAS:** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)

            **NOX:** Nitric oxide concentration (parts per 10 million)

            **RM:** Average number of rooms per dwelling

            **AGE:** Proportion of owner-occupied units built prior to 1940

            **DIS:** Weighted distances to five Boston employment centers

            **RAD:** Index of accessibility to radial highways

            **TAX:** Full-value property tax rate per $10,000

            **PTRATIO:** Pupil-teacher ratio by town

            **B:** 1000(Bk — 0.63)², where Bk is the proportion of people of African American descent by town

            **LSTAT:** Percentage of lower status of the population

            **MEDV:** Median value of owner-occupied homes in $1000s
            """
                ),
            ],
            style={
                "width": "30%",
                "display": "inline-block",
                "backgroundColor": colors["background"],
            },
        ),
        html.Div(
            [dcc.Graph(id="boston-housing-data")],
            style={"width": "68%", "float": "right", "display": "inline-block"},
        ),
    ]
)


@app.callback(
    dash.dependencies.Output("boston-housing-data", "figure"),
    [
        dash.dependencies.Input("xaxis-dropdown", "value"),
        dash.dependencies.Input("yaxis-dropdown", "value"),
    ],
)
def update_figure(xaxis_dropdown, yaxis_dropdown):
    return {
        "data": [
            go.Scatter(
                x=df[xaxis_dropdown],
                y=df[yaxis_dropdown],
                mode="markers",
                opacity=0.7,
                marker={
                    "size": 7,
                    "line": {"width": 0.5, "color": "white"},
                    "color": "black",
                },
            )
        ],
        "layout": go.Layout(
            xaxis={"title": xaxis_dropdown},
            yaxis={"title": yaxis_dropdown},
            margin={"l": 40, "b": 40, "t": 10, "r": 10},
            legend={"x": 0, "y": 1},
            hovermode="closest",
            plot_bgcolor=colors["background"],
            paper_bgcolor=colors["background"],
        ),
    }


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=False, port=8050)
