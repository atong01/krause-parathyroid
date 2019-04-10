import json
import numpy as np
import pickle
import gzip

import matplotlib
matplotlib.use('Agg')
from matplotlib import cm

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

from flask import Flask

# init app
app_name = 'parathyroid-phate-magic'
data_name = "PPA"
server = Flask(app_name)
app = dash.Dash(app_name, server=server, csrf_protect=False)
port = 8901
title = 'Primary Parathyroid Adenoma'

# Load embeddings
phate2d_mat = np.load("{}Phate2d.npy".format(data_name))
phate3d_mat = np.load("{}Phate3d.npy".format(data_name))

# Load magic
def load(filename):
    f = gzip.GzipFile(filename, 'rb')
    return pickle.load(f)

mg = load('magic.pickle.gz')

# Load gene names
gene_names = np.sort(mg.X.columns).astype(np.str).tolist()

# init gene expression color data
starting_gene = 'PTH (ENSG00000152266)'
gene_data = mg.transform(genes=starting_gene).to_dense().values.flatten()

# misc plot settings
bg_color = '#EEE'
default_camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0.02, y=-0.1, z=-0.11),
    eye=dict(x=1.46, y=1.71, z=0.05)
)

# generate viridis color scale (https://plot.ly/python/matplotlib-colorscales/)
viridis_cmap = cm.viridis
viridis_rgb = []
norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
for i in range(0, 255):
    k = matplotlib.colors.colorConverter.to_rgb(viridis_cmap(norm(i)))
    viridis_rgb.append(k)


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []
    for k in range(pl_entries):
        C = (np.array(cmap(k * h)[:3]) * 255).astype(np.uint8)
        pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])
    return pl_colorscale

viridis = matplotlib_to_plotly(viridis_cmap, 255)

# Plotting function


def scatter_plot_3d(
        x=phate3d_mat[:, 0], y=phate3d_mat[:, 1], z=phate3d_mat[:, 2],
        xlabel='PHATE1', ylabel='PHATE2', zlabel='PHATE3',
        color=gene_data,
        size=1,
        camera=default_camera,
        plot_type='scatter3d'
):
    def axis_template_3d(title):
        return dict(
            showbackground=True,
            backgroundcolor=bg_color,
            title=title,
            titlefont=dict(
                color='#000', family='Open Sans, sans-serif', size=12),
            type='linear',
            showticklabels=False,
            showgrid=False,
            showspikes=False,
            zeroline=False,
            color='#FFF',
        )

    def axis_template_2d(title):
        return dict(
            xgap=10, ygap=10,
            backgroundcolor=bg_color,
            title=title,
            titlefont=dict(
                color='#000', family='Open Sans, sans-serif', size=12),
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            color='#BBB',
        )
    data = [dict(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=size,
            opacity=1,
            colorscale=viridis,
            line=dict(color='#BBB'),
            color=color,
            colorbar=dict(
                title='expression level',
                titlefont=dict(size=12, color='#000',
                               family='Open Sans, sans-serif'),
                tickfont=dict(color='#000', family='Open Sans, sans-serif'),
                titleside='right',
                thickness=12,
                len=0.8,
                x=1,
                xanchor='left',
                xpad=10
            )
        ),
        type=plot_type
    )]
    layout = dict(
        font=dict(family='Open Sans', color='#BBB'),
        mode='none',
        hovermode=False,
        width=400,
        height=400,
        margin=dict(r=0, t=0, l=0, b=0),
        showlegend=False,
        scene=dict(
            xaxis=axis_template_3d(xlabel),
            yaxis=axis_template_3d(ylabel),
            zaxis=axis_template_3d(zlabel),
            aspectmode='cube',
            hovermode=False,
            camera=camera
        )
    )
    if plot_type == 'scatter':
        layout['xaxis'] = axis_template_2d(xlabel)
        layout['yaxis'] = axis_template_2d(ylabel)
        layout['plot_bgcolor'] = bg_color
        layout['autosize'] = False
        layout['width'] = 300
        layout['height'] = 250
        layout['margin'] = dict(r=0, t=20, l=20, b=20)
        del layout['scene']
        del data[0]['z']
    else:
        data[0]['marker']['showscale'] = False
    return dict(data=data, layout=layout)

figure3d = scatter_plot_3d()
figure2d = scatter_plot_3d(
    x=phate2d_mat[:, 0],
    y=phate2d_mat[:, 1],
    plot_type='scatter',
    size=1.5
)

# HTML layout ------------------------------------------------------------

app.title = title
app.layout = html.Div([
    # Title
    # html.H2(title),
    # Graphs
    html.Div([dcc.Graph(id='graph-3d', figure=figure3d)],
             style={'float': 'left', 'width': '400px'}),
    html.Div([dcc.Graph(id='graph-2d', figure=figure2d)],
             style={'margin': '60px 0px 40px 20px', 'float': 'left',
                    'width': '280', 'height': '300px'}),
    # Gene select
    html.Div([
        html.P("Select a gene:",
               style={'display': 'inline-block', 'vertical-align': 'middle',
                      'font-size': '2.5rem'
                      }
               ),
        html.Div([
            dcc.Dropdown(id='gene-dropdown', value=starting_gene,
                         options=[{'label': i, 'value': i} for i in gene_names]
                         )],
                 id='gene-dropdown-div',
                 style={'width': '400px', 'display': 'inline-block',
                        'margin-left': '10px'
                        }
                 )
    ], style={'clear': 'both', 'margin-top': '20px'}),
    # Attribution
    html.Div([
        html.P('Created by Scott Gigante and Kristina Yim',
               style={'color': '#CCC'})
    ]),
    # Hidden elements for storing values
    html.Div(id='camera-data', style={'display': 'none'}),
    html.Div(id='gene-data', style={'display': 'none'})
], className='container')

# UI functions -----------------------------------------------------------

# recover gene expression


@app.callback(
    Output('gene-data', 'children'),
    [Input('gene-dropdown', 'value')])
def recover_expression(selected_gene):
    gene_data = mg.transform(genes=[selected_gene]).to_dense().values.flatten()
    # return base64.b64encode(gene_data)
    return gene_data


# retrieve camera upon gene-dropdown-click ?
@app.callback(
    Output('camera-data', 'children'),
    [Input('gene-dropdown', 'value')],
    [State('graph-3d', 'figure')])
def test_function(n_clicks, figure):
    camera = figure['layout']['scene']['camera']
    return json.dumps(camera)

# update figures


@app.callback(
    Output('graph-3d', 'figure'),
    [Input('gene-data', 'children'),
     Input('camera-data', 'children')])
def update_figure(gene_data, camera_data):
    #gene_data = base64.decodestring(gene_data_encoded)
    #gene_data = np.frombuffer(gene_data, dtype=np.float16)
    camera = json.loads(camera_data)
    return scatter_plot_3d(
        color=gene_data,
        camera=camera,
        plot_type='scatter3d'
    )


@app.callback(
    Output('graph-2d', 'figure'),
    [Input('gene-data', 'children')])
def update_figure(gene_data):
    # gene_data = base64.decodestring(gene_data_encoded)
    # gene_data = np.frombuffer(gene_data, dtype=np.float16)
    return scatter_plot_3d(
        color=gene_data,
        x=phate2d_mat[:, 0],
        y=phate2d_mat[:, 1],
        plot_type='scatter',
        size=1.5
    )

external_css = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "//fonts.googleapis.com/css?family=Open+Sans:300,400",
                "https://codepen.io/chriddyp/pen/brPBPO.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

# for publishing on Heroku
app.component_suites = [
    'dash_core_components',
    'dash_html_components'
]

if __name__ == '__main__':
    app.run_server(port=port, debug=True)
