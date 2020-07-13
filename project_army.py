'''
Author :- SIDHHANT BHATNAGAR

some instructions:-
--please make sure the code and all other supported files are in the same folder.
--please make sure that the background picture is in a folder named "assets" (dash recognises "assets")

 JAI HIND !!
 
'''
#------------------------------------------------------------------------------------------

import threading
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import webbrowser
from dash.dependencies import Output, Input
import plotly.express as px
from scipy.interpolate import interp1d

app = dash.Dash()

# --------------------------------------------------------------------------------------
# Import and clean the CSV data using pandas
dataset = pd.read_csv('C:/Users/sidhh/Desktop/Military_project/GTD_new.csv', encoding="ISO-8859-1")

dataset.isnull().any(axis=0)
dataset['city'].fillna('Unknown', inplace=True)


# --------------------------------------------------------------------------------------
# Initial Setup

map_value = px.scatter_mapbox(dataset, lat="latitude", lon="longitude", hover_name="country_txt",
                              mapbox_style="carto-darkmatter",
                              hover_data=["city", "attacktype1_txt","nkill"],color_discrete_sequence=["fuchsia"], zoom=2, height=600)

graph1_value = {'data': [
    go.Scatter(
        x=np.array(dataset['year'].unique()),
        y=np.array(dataset['attacktype1_txt'].values),
        mode='markers',
        marker={
            'size': 12,
            'line': {'width': 0.5, 'color': 'indianred'}})],
    'layout': go.Layout(
        title="MAXIMUM OCCURRING ATTACKS EACH YEAR ACROSS THE GLOBE",
        font=dict(family='Arial Black', size=10, color='orange'),
        margin={'l': 120, 'b': 40, 'r': 100},
        plot_bgcolor="#111111",
        paper_bgcolor="#111111"
        )}

pv = pd.pivot_table(dataset, index=['region_txt'], values=['nkill'], aggfunc=sum, fill_value=0)
graph2_value = {
    'data': [go.Bar(x=pv.index, y=pv['nkill'],marker_color='indianred')],
    "layout": {
        "title": "NUMBER OF CASUALTIES FROM 1970-2018",
        "plot_bgcolor":"#111111",
        "paper_bgcolor":"#111111",
        'font': {
            'color': '#7FDBFF',
            'family': 'Arial Black',
            'size': 10,
        },'margin':{'l': 80, 'b': 150, 'r': 80}
    }}

df=dataset.groupby(['country_txt']).transform('count')
m=zip(df['country'],dataset['latitude'],dataset['longitude'],dataset['country_txt'],dataset['city'])
data_new=pd.DataFrame(m, columns=['country','lat','lon','Country','City'])
l1=data_new.country.values.tolist()
q=interp1d([1,max(l1)],[5,10])
r=q(l1)

pv2 = pd.pivot_table(dataset, index=['year'], values=['nkill'], aggfunc=sum, fill_value=0)

# --------------------------------------------------------------------------------------
# Create App Layout

app.layout = html.Div(
   className='row',
    style={
     'padding':'0',
      'margin':'0',
      'background-image':'url("/assets/army.jpg")',
      'background-position':'top',
      'background':'cover',
      'opacity':'0.8',
       'position': 'absolute', 
       'top': '0', 
       'left': '0', 
      'right':'0'},
  children=[
    html.H1(children="ANALYSIS",
            style={
                'textAlign': 'center',
                'color': 'black',
                'font-family':'Times',
                'font-size':'320%',
                'margin-top':'3%',
            }),

    html.Div([
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': 'Select a country', 'value': "None",}] + [{'label': i, 'value': i} for i in
                                                                        dataset.country_txt.unique()],
            placeholder='Select a country',
            value='None',
            style={'font-family':'sans-serif','font-weight':'bold','width': '250px', 'margin-top': '10px','color':'darkgreen', }
        ),
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label': 'Select a region', 'value': "None"}] + [{'label': i, 'value': i} for i in
                                                                       dataset.region_txt.unique()],
            placeholder='Select a region',
            value='None',
            style={'font-family':'sans-serif','border-radius':'5px','font-weight':'bold','width': '250px', 'margin-top': '10px','color':'darkgreen', }
        ),
        dcc.Dropdown(
            id='attack-type-dropdown',
            options=[{'label': 'Select attack type', 'value': "None"}] + [{'label': i, 'value': i} for i in
                                                                          dataset.attacktype1_txt.unique()],
            placeholder='Select attack type',
            value='None',
            style={'font-family':'sans-serif','border-radius':'5px','font-weight':'bold','width': '250px', 'margin-top': '10px','color':'darkgreen',}
        ),
        dcc.Dropdown(
            id='day-dropdown',
            options=[{'label': 'Select Day', 'value': "None"}] + [{'label': i, 'value': i} for i in
                                                                  sorted(dataset.day.unique())],
            placeholder='Select Day',
            value='None',
            style={'font-family':'sans-serif','border-radius':'5px','font-weight':'bold','width': '250px', 'margin-top': '10px','color':'darkgreen' ,}
        ),
        dcc.Dropdown(
            id='month-dropdown',
            options=[{'label': 'Select Month', 'value': "None"}] + [{'label': i, 'value': i} for i in
                                              sorted(dataset.month.unique())],
            placeholder='Select Month',
            value='None',
            style={'font-family':'sans-serif','border-radius':'5px','font-weight':'bold','width': '250px', 'margin-top': '10px','color':'darkgreen' ,}
        ),
    ], style={'width': '60%', 'margin': 'auto', 'display': 'flex', 'flex-flow': 'wrap',
              'justify-content': 'space-around', 'margin-bottom': '20px' }),

    # Button
    html.Div([
        html.Button(
            'EXECUTE',
            id="filter-button",
            style={'outline':'true', 'color':'black','font-family':'sans-serif','width': '100px','font-weight':'bold', 'border-radius':'5px','height': '30px' , 'margin-bottom': '10px'},
            n_clicks=0,
        )
    ], style={'width': '100%', 'text-align': "center", 'margin-top': '25px', 'margin-bottom': '40px'}),

html.Div([
       dcc.Graph(
            style={"border":'2px solid black',"padding":'0', "width": '80%', 'height': '40%', "margin-left": '12%','margin-bottom': '10%','margin-top': '5%'},
            id='map',figure=map_value)
    ]),
  

    
    html.Div([
        dcc.Graph(style={'width': '90%', 'height': '400px', 'margin-top':'-70px','margin-left':'6%' , 'border':'2px solid white' ,},
                  id='scatter-chart', figure=graph1_value)
    ]),
   
    html.Br(),
    html.Div([
        
        dcc.Graph(
            style={"margin":'4%',"border":"2px solid black","width": '93%', 'height': '50%', "margin-left": '5%', "padding":'0'},
            id='heat_map',figure=px.density_mapbox(data_new,lat='lat',lon='lon',radius=r,zoom=3,mapbox_style='stamen-watercolor',hover_data=['Country','City'],
                                                   title="HEAT MAP DEPICTING INTENSITY OF ATTACKS ALL OVER THE GLOBE"))]),

   
    html.Div(        children=[
            dcc.Graph(style={'border':'2px solid white', 'padding':'0','margin':'5%',"width": '80%', 'height': '50%', "margin-left": '10%','margin-bottom': '50px', "padding":'0'},
                      id='casualties-graph', figure=graph2_value)
        ]
    ),
    
    html.Br(),
    
    html.Div(children=[
     html.H1(children="MAKE YOUR SELECTION",
            style={
                'margin-left':'28%',
                'color': 'blue',
                'font-family':'sans-serif',
                'font-size':'120%',
                'text-align':'center',
                'box-sizing':'border-box',
                'border':'2px solid black',
                'width':'40%',
                'padding':'1%',
                'background': 'linear-gradient(to right top, orange, white 50%, green )',
                
                },
            ),
        
        dcc.Dropdown(
            id='my_dropdown',
          
            options=[{'label': 'Attack Type', 'value': 'attacktype1_txt'},
                     {'label': 'Target Type', 'value': 'targtype1_txt'},
                     
                     {'label': 'Weapon Type', 'value': 'weaptype1_txt'}],
        
            placeholder='Select',
            value='targtype1_txt',
            style={'font-family':'sans-serif','font-weight':'bold','width': '400px', 'margin-top':'20px', 'margin-bottom':'15px','margin-left':'27%','text-align':'center',}
            
        )]),
     
    
html.Div([
    dcc.Graph(style={ "background-color":"black","width":'80%','border':'2px solid black','margin-top':'-2px', "margin-left":'10%',},id='the_graph')
    ]),
html.Br(),

html.Div([
    dcc.Graph(id='timeseries',
          config={'displayModeBar': False},
          style={"border":'2px solid white','width':'80%','height':'40%','margin-left':'10%', "margin-bottom":'5%',},
          animate=True,
          figure=px.line(pv2,
                         x=pv2.index,
                         y=pv2['nkill'],
                         labels={'year':'YEAR','nkill':'DEATHS'},
                         title="DEATH PATTERN",
                         template='plotly_dark').update_layout(
                                   {'plot_bgcolor': '#111111',
                                    'paper_bgcolor': '#111111',
                                    'font': {
            'color': 'red',
            'family': 'Arial Black',
            'size': 10}}
            ))])
                            
])


# --------------------------------------------------------------------------------------
# Callbacks

@app.callback(
    Output(component_id='map', component_property='figure'),
    [Input('filter-button', 'n_clicks'),
     Input(component_id='country-dropdown', component_property='value'),
     Input(component_id='region-dropdown', component_property='value'),
     Input(component_id='attack-type-dropdown', component_property='value'),
     Input(component_id='day-dropdown', component_property='value'),
     Input(component_id='month-dropdown', component_property='value')]
)
def update_graph(click, country, region, attack_type, day, month):
    dff = dataset.copy()
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'filter-button' not in changed_id:
        # button was not clicked show previous data
        return map_value

    if country != 'None':
        dff = dff[dff["country_txt"] == country]
    if region != 'None':
        dff = dff[dff["region_txt"] == region]
    if attack_type != 'None':
        dff = dff[dff["attacktype1_txt"] == attack_type]
    if day != 'None':
        dff = dff[dff["day"] == day]
    if month != 'None':
        dff = dff[dff["month"] == month]

    return px.scatter_mapbox(dff, lat="latitude", lon="longitude", hover_name="country_txt",
                              mapbox_style='stamen-terrain',
                              hover_data=["city", "attacktype1_txt","nkill"],color_discrete_sequence=["fuchsia"], zoom=3, height=600
                              )
          
@app.callback(
    Output(component_id='the_graph', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)

def update_graph(my_dropdown):
    dff = dataset.iloc[0:10000,[11,13,16]]

    piechart=px.pie(
            data_frame=dff,
            names=my_dropdown,
            labels={'targtype1_txt':'Target','weaptype1_txt':'Weapon','attacktype1_txt':'Attack'})
    
    piechart.update_traces(textposition='inside')
    piechart.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    piechart.update_traces(textfont_size=10,
                  marker=dict(line=dict(color='#000000', width=0.5)))

    return (piechart)


def open_browser():
    
    webbrowser.open("http://127.0.0.1:4050", new=1, autoraise=True)


if __name__ == '__main__':
      x = threading.Thread(target=open_browser(), args=(1,))
      x.start()
      app.run_server(port=4050)

##########################################################################################################################