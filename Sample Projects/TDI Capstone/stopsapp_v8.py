# -*- coding: utf-8 -*-
"""

@author: m-askariyeh

"""
#####################################################################################
# This file contains what i did for TDI capstone project.
# It reads the data and runs a web application which gives the option to filter the data, and visualize it.
# It shows the location of parked vehicles in Houston and Dallas.
# Practitioners can select different models (8 GM models), months (July and October 2019), and time periods (night time and day time).
# At first step, it shows parking locations and barcharts indicateingthree measures: number of parked vehicles, average parking time, and sum of the parking time for filetered data.
# Moreover, it sorts and provide 10 zipcodes with highest sum of parked vehicles.
# At second step, practitioners can define the lat and lon of a particular point and also a radius in miles and analyze the data for the points that fall within the defined circle.
# It also provides the ranking of the sum of parking time within this circle in comparison with all zipcodes.      
#####################################################################################

### Importing packages
import pandas as pd
import numpy as np
import plotly.express as px
from flask import Flask
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import geopy.distance


from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2, r):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 3959* c
    if km < r:
        return 1
    return 0



def two_point_distance(lat, lon, center_lat, center_lon, radius):
    coords = (lat, lon)
    center = (center_lat, center_lon)
    if  geopy.distance.geodesic(coords, center).miles <= radius:
        return 1
    return 0
    


# Setting up the server
mapbox_access_token = 'pk.eyJ1IjoiYXNrYXJpeWVoIiwiYSI6ImNrdW1xZzhyYTN1Zm8yd21hY2Z6OWp1eGEifQ.gWgug3D4KdvxOXsvEP86ug'
server = Flask('my_app')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# =============================================================================

percentage = 0

boundaries = 1.5
areas = ['HGAC','NCTCOG'] # 'HGAC', 
periods = ['July', 'October'] # 'July', 

nighttime_period = [i for i in range(6, 12)]
night_length = 3600

print('Reading the input data ...')
master = {}
for i in areas:
    for j in periods:
        path = f'D:/stopsdata/data/{i}_{j}_mod.csv'
        master[i,j] = pd.read_csv(path)
        master[i,j]['day_night'] = 'dtime'
        master[i,j].loc[(master[i,j]['Hour'].isin(nighttime_period)) & (master[i,j]['soakTime'] > night_length), ['day_night']]= 'ntime'
    
print('Just read the input set!')

areatitle = {'HGAC' : 'Houston-Galveston Area',
             'NCTCOG' : 'Dallas-Fort Worth Area'}

# =============================================================================

# =============================================================================

def zoom_center(lons: tuple=None, lats: tuple=None, lonlats: tuple=None,
        format: str='lonlat', projection: str='mercator',
        width_to_height: float=2.0) -> (float, dict):
    """Finds optimal zoom and centering for a plotly mapbox.
    Must be passed (lons & lats) or lonlats.
    Temporary solution awaiting official implementation, see:
    https://github.com/plotly/plotly.js/issues/3434
    
    Parameters
    --------
    lons: tuple, optional, longitude component of each location
    lats: tuple, optional, latitude component of each location
    lonlats: tuple, optional, gps locations
    format: str, specifying the order of longitud and latitude dimensions,
        expected values: 'lonlat' or 'latlon', only used if passed lonlats
    projection: str, only accepting 'mercator' at the moment,
        raises `NotImplementedError` if other is passed
    width_to_height: float, expected ratio of final graph's with to height,
        used to select the constrained axis.
    
    Returns
    --------
    zoom: float, from 1 to 20
    center: dict, gps position with 'lon' and 'lat' keys

    >>> print(zoom_center((-109.031387, -103.385460),
    ...     (25.587101, 31.784620)))
    (5.75, {'lon': -106.208423, 'lat': 28.685861})
    """
    if lons is None and lats is None:
        if isinstance(lonlats, tuple):
            lons, lats = zip(*lonlats)
        else:
            raise ValueError(
                'Must pass lons & lats or lonlats'
            )
    
    maxlon, minlon = max(lons), min(lons)
    maxlat, minlat = max(lats), min(lats)
    center = {
        'lon': round((maxlon + minlon) / 2, 6),
        'lat': round((maxlat + minlat) / 2, 6)
    }
    
    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array([
        0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096,
        0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568,
        47.5136, 98.304, 190.0544, 360.0
    ])
    
    if projection == 'mercator':
        margin = 1.2
        height = (maxlat - minlat) * margin * width_to_height
        width = (maxlon - minlon) * margin
        lon_zoom = np.interp(width , lon_zoom_range, range(20, 0, -1))
        lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
        zoom = round(min(lon_zoom, lat_zoom), 2)
    else:
        raise NotImplementedError(
            f'{projection} projection is not implemented'
        )
    
    return zoom, center
# =============================================================================

app = dash.Dash( __name__,
                server=server,
                external_stylesheets=external_stylesheets,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                suppress_callback_exceptions=False)

app.layout = html.Div([ ## whole page
    html.Div([ ## Left panel
         html.Img(
             src= app.get_asset_url("tdi_logo.png"),
             alt="TDI logo",
             width="210",
             height="40"
             ),
         
         html.H6("TDI Capstone Project",
                 style={'width': "60%",
                        'font-weight': 'bold'}),
         
         html.H6("developed by:",
                 style={'width': "40%",
                        'fontSize':15,}),         
         
         html.Div([
             html.A("Mohammad H. Askariyeh",
                    href="https://www.linkedin.com/in/mhaskariyeh/",
                    target="_blank",
                    style={'fontSize':15,
                           # 'font-style': 'italic',
                           'font-weight': 'bold',
                           'color': 'black',
                           'justify-content': 'left',
                           'verticalAlign': 'middle',
                           'textAlign': 'left',
                           'alignItems': 'left',
                           }
                    ),
             
             ],
             style={'width':'100%',
                    'height': '20px',
                    'verticalAlign': 'middle'}
                 ),
                  
         html.Br(),

       
         # Radio botton
         html.Div(
             [
             html.H6("Area:", 
                     style={'width': "150px",
                                  'display': 'inline-block',
                                  'font-weight': 'bold',},
                     className="radio_button"),
             dcc.RadioItems(
                 options=[
                     {"label": "Dallas-Fort Worth (NCTCOG)", "value": "NCTCOG"},
                     {"label": "Houston (HGAC)", "value": "HGAC"},
                     ],
                 id="area",
                 value = "NCTCOG",
                 labelClassName="label__option",
                 inputClassName="input__option",
                 # labelStyle={'display': 'inline-block'}
                 ),
             ],
             className="radio_button",
             ),


         html.Div([ ## Vehicle Model options
                   html.H6("Vehicle Model:",
                           style={'width': "150px",
                                  'font-weight': 'bold',
                                  'display': 'inline-block',
                                  'lineHeight': '30px',
                                  'height':'30px',
                                  'verticalAlign': 'middle'}),
 
                   
                   
                             # Show vehicles model checkbox
                  dcc.Checklist(
                      id='model',
                      className="checkbox",
                      options=[
                          {'label': 'Bolt_EV', 'value': 'BOLT_EV'},
                          {'label': 'Cruze', 'value': 'CRUZE'},
                          {'label': 'CTS', 'value': 'CTS'},
                          {'label': 'Enclave', 'value': 'ENCLAVE'},
                          {'label': 'Encore', 'value': 'ENCORE'},
                          {'label': 'Malibu', 'value': 'MALIBU'},
                          {'label': 'Volt', 'value': 'VOLT'},
                          {'label': 'XT5', 'value': 'XT5'}
                          ],
                      value= ['BOLT_EV', 'VOLT'],
                      ),
                   

                     ],
                  style={'verticalAlign': 'middle',
                         }
                  #className = "dropdown"
                  ),
                           

         html.Div([ ## Third Drop Down
                   html.H6("Dates:",
                           style={'width': "150px",
                                  'display': 'inline-block',
                                  'font-weight': 'bold'}
                           ),
                   dcc.Dropdown(id = "period",
                                multi=False,
                                placeholder="Month/Year",
                                options=[{'label': 'July 2019', 'value': 'July'},
                                         {'label': 'Oct 2019', 'value': 'October'}],
                                value= 'October',
                                # type='text',
                                style={'width': "44%",
                                       'lineHeight': '30px',
                                       'height':'30px',
                                       'display': 'inline-block',
                                       'verticalAlign': 'middle'})
                   ],
                  #className = "dropdown"
                  style={'verticalAlign': 'middle',
                         'margin': '1px',
                         }
                  ),
         
         

         html.Div([ ## Vehicle Model options
                   html.H6("Time Period:",
                           style={'width': "150px",
                                  'font-weight': 'bold',
                                  'display': 'inline-block',
                                  'lineHeight': '30px',
                                  'height':'30px',
                                  'verticalAlign': 'middle'}),
 
                   
                   
                             # Show dnperiod checkbox
                  dcc.Checklist(
                      id='dnperiod',
                      className="checkbox",
                      options=[
                          {'label': 'Day Time', 'value': 'dtime'},
                          {'label': 'Night Time', 'value': 'ntime'},

                          ],
                      value= ['dtime', 'ntime'],
                      ),
                   

                     ],
                  style={'verticalAlign': 'middle',
                         }
                  #className = "dropdown"
                  ),
                           
    

         html.Div([ ## First Drop Down
                   html.H6("Min Parking Time (min):",
                           style={'width': "150px",
                                  'display': 'inline-block',
                                  'font-weight': 'bold',}),
                   dcc.Input(id = 'cutoff',
                             type = "number",
                             placeholder = "cutoff",
                             min = 10, 
                             max = 120, 
                             step = 5,
                             value= 30,
                             size = "20",
                             autoComplete = "off",
                             style={'width': "20%",
                                    'display': 'inline-block',
                                    'lineHeight': '20px',
                                    'height':'30px' },)],
                  style={'verticalAlign': 'middle'},
                  ), 
         

          html.Div([html.Button(children=['Filter and Visualize'],
                                id='submit_button',
                                n_clicks=0,
                                className = 'profile_button'),
              ],
              style={'width':'100%',
                     'height': '80px',
                     'verticalAlign': 'top'}
                  ),



######## Analysis panel #######################################################
         html.Div([
             html.Div([html.H6("Point Evaluation",
                               style={'width': "180px",
                                      'display': 'inline-block',
                                      'font-weight': 'bold',},
                               ),
                       ],),
             
             html.Div([ ## Gets latitude input
                       html.H6("Latitude:",
                               style={'width': "150px",
                                      'display': 'inline-block',
                                      'font-weight': 'bold',}),
                       dcc.Input(id = 'lat_inp',
                                 type = "number",
                                 placeholder = "Latitude",
                                 step = 0.005,
                                 size = "20",
                                 autoComplete = "off",
                                 style={'width': "20%",
                                        'display': 'inline-block',
                                        'lineHeight': '20px',
                                        'height':'30px' },)],
                      style={'verticalAlign': 'middle'},
                      ),          
    
             html.Div([ ## Gets longitude input
                       html.H6("Longitude:",
                               style={'width': "150px",
                                      'display': 'inline-block',
                                      'font-weight': 'bold',}),
                       dcc.Input(id = 'lon_inp',
                                 type = "number",
                                 placeholder = "Longitude",
                                 step = 0.005,
                                 size = "20",
                                 autoComplete = "off",
                                 style={'width': "20%",
                                        'display': 'inline-block',
                                        'lineHeight': '20px',
                                        'height':'30px' },)],
                      style={'verticalAlign': 'middle'},
                      ),
    
    
             html.Div([ ## Gets radius input
                       html.H6("Radius (mile):",
                               style={'width': "150px",
                                      'display': 'inline-block',
                                      'font-weight': 'bold',}),
                       dcc.Input(id = 'radius_inp',
                                 type = "number",
                                 placeholder = "Radius",
                                 min = 1, 
                                 max = 50, 
                                 step = 1,
                                 value= 4,
                                 size = "20",
                                 autoComplete = "off",
                                 style={'width': "20%",
                                        'display': 'inline-block',
                                        'lineHeight': '20px',
                                        'height':'30px' },)],
                      style={'verticalAlign': 'middle'},
                      ),
    
    
    
             html.Div([
                 html.Button(children=['Evaluate'],
                             id='eval_button',
                             n_clicks=0,
                             className = 'profile_button'),
                 ],
                 style={'width':'100%',
                        'height': '150px',
                        'verticalAlign': 'top'}
                     ),
             ],
             id="evaluation_pan")


         
		], className = "split left"),

    
	html.Div([ ##  Right panel
           
			html.Div([
                html.H1("Evaluation of Electric Vehicles Charging Station Placement",
                        style={'width': "100%",
                               'height':'2%',
                               'verticalAlign': 'middle',
                               'lineHeight': '4%',
                               'font-weight': 'bold',})
                ],
                style={'width':'100%',
                    'height': '5%',
                    'verticalAlign': 'middle'}
                ),

			dcc.Loading([dcc.Graph(id = 'graph'),
                # html.Div(id='errormessage'),
                dcc.Markdown(id='errormessage')]),
            
            html.Div(
                [
                    html.Button("Export Data", 
                                id="export_button", 
                                className = 'sec_type_button'),
                    dcc.Download(id="download-dataframe-csv"),
                    dcc.Store(id='dfout')
                    ],
                ),
            dcc.Loading([html.Div(id='evalmessage')]),                     
            ], 
          className = "split right")
    ])

@app.callback(
    [#Output("Radio1", 'style'),
     Output('graph', 'style'),
     Output('graph', 'figure'),
     Output('errormessage', 'children'),
     Output('lat_inp', 'min'),
     Output('lat_inp', 'max'),
     Output('lat_inp', 'value'),
     Output('lon_inp', 'min'),
     Output('lon_inp', 'max'),
     Output('lon_inp', 'value'),
     # Output('evalmessage', 'children'),
     Output('dfout', 'data')],
    [Input('submit_button', "n_clicks"),
     Input('eval_button', "n_clicks")],
    
    [State('cutoff', 'value'),
    State('period', 'value'),
    State('area', 'value'),
    State('model', 'value'),
    State('dnperiod', 'value'),
    State('graph', 'figure'),
    State('lat_inp', 'min'),
    State('lat_inp', 'max'),
    State('lat_inp', 'value'),
    State('lon_inp', 'min'),
    State('lon_inp', 'max'),
    State('lon_inp', 'value'),
    State('radius_inp', 'value'),
    State('dfout', 'data')
    ]
    )

  
def update_graph(submit_button,
                 eval_button,
                 cutoff,
                 period,
                 area,
                 model,
                 dnperiod,
                 graph_content,
                 lat_min,
                 lat_max,
                 lat_value,
                 lon_min,
                 lon_max,
                 lon_value,
                 radius_value,
                 outputexport
                 ):
 
    # =============================================================================
    first_page = '''

            ##        
            #### **Background:**
            * Importance of climate change, global warming, and air pollution reduction
            * Significant contribution of transportation sector to air pollution in urban areas
            * Vehicles with alternative fuels like electric vehicles offer an opportunity to address many important issues 
            * Lack of enough charging stations is a problem to increase electric vehicles adoption rate
            * A great opportunity for charging vehicles is when they are parked
            ##
            ###
            ####                             
            #### **Methodology:**
            * Developed an application and used the count and parking time of vehicles to identify hotspots in urban areas
            * Used more than 20 million records indicating location and parking time of vehicles
            * Goal: To identify the areas with relatively higher potentials to place charging stations

    '''    
    # =============================================================================
    
    ctx = dash.callback_context
    df_org = []
    df55=[]
    df_top_zips = []
    outputexport = pd.DataFrame().to_dict('records')
    if not ctx.triggered:
        evalmessage = 'No Evaluation Yet.'
        error_message = first_page # 'Ready!'
        display =  {'display': 'none'}
        fig = {}
        lat_min = 0
        lat_max = 0
        lat_value = 0
        
        lon_min = 0
        lon_max = 0
        lon_value = 0
        outputexport = pd.DataFrame().to_dict('records')
    else:
        clicked_button = ctx.triggered[0]['prop_id'].split('.')[0]
        display = {'display': 'flex', 'justify-content': 'center'}
        evalmessage = 'No Evaluation Yet.'
        error_message = first_page # 'Ready!'
        
        outputexport= None
        
        if submit_button > 0:
              
            # ============================================================================= 
            # Data prep
            
            display = {'display': 'flex', 'justify-content': 'center'}
           
            print(f'area = {area}')
            print(f'period = {period}')
            print(f'cutoff = {cutoff}')
            print(model)
            print(dnperiod)
            # print(dnperiod[0])
             # period1 = 'July'
            # print(f'period1 = {period1}')
            
            # area1 = 'HGAC'
            # print(f'area1 = {area1}')
            # df1 = master['HGAC', 'October']
            # 'HGAC', 'October'  'NCTCOG', 'July'
            
            df1 = master[area, period]
            # print(df1.columns)
            # df1 = df1[['latitude', 'longitude', 'postalCode', 'Date', 'Hour', 'make', 'model', 'year', 'soakTime']]
            
            # df1['day_night'] = 'dtime'
            # nighttime_period = [i for i in range(6, 11)]
            # df1.loc[(df1['Hour'].isin(nighttime_period)) & (df1['soakTime'] > 3600), ['day_night']]= 'ntime'
           
            df = df1[df1['soakTime']> cutoff]
            
            if len(dnperiod) == 1:
                df = df[df['day_night'] == dnperiod[0]]
                
            # print(df.columns)
    
            # model = ['BOLT_EV','VOLT'] # 
            # print(nighttime_period)
            df = df[df['model'].isin(model)]
            
            df['postalCode'] = df['postalCode'].astype(int)
            df['soakTime'] = df['soakTime'].astype(int)
            
            df_org = df 
            
            if 'eval_button' in clicked_button:
                lat_min = lat_min
                lat_max = lat_max
                lat_value = lat_value
                lon_min = lon_min
                lon_max = lon_max
                lon_value = lon_value
                outputexport = outputexport
                
                df['in_boundary'] = df.apply(lambda row: two_point_distance(row['latitude'],
                                                                            row['longitude'],
                                                                            lat_value,
                                                                            lon_value,
                                                                            radius_value),
                                             axis=1)
# =============================================================================
# =============================================================================
                
                print(f'before {df.shape}')
                df = df[df['in_boundary']==1]
                print(f'after {df.shape}')
                
                R = radius_value
                center_lon = lon_value
                center_lat = lat_value
                t = np.linspace(0, 2*np.pi, 100)
                circle_lon = center_lon + R/54.6*np.cos(t)
                circle_lat = center_lat +  R/69.39*np.sin(t)
                       
                coords=[]
                for lo, la in zip(list(circle_lon), list(circle_lat)):
                    coords.append([lo, la]) 
                
                layers=[dict(sourcetype = 'geojson',
                             source={ "type": "Feature",
                                     "geometry": {"type": "LineString",
                                                  "coordinates": coords
                                                  }
                                    },
                             color=   'red',
                             type = 'line',   
                             line=dict(width=1.5)
                            )]
            
           
            
            
            outputexport= df.to_dict('records')

            # =============================================================================         
            # Sort the zipcodes based on original sum of the parking times
            
            df7 = df_org.groupby(['postalCode'])['soakTime'].sum().reset_index()
            df77 = df7.sort_values(by = ['soakTime'], ascending=False).reset_index()
            df_top_zips_77 = df77['postalCode'].head(10)
            print(df_top_zips_77)
    
            # =============================================================================         
            # Sort the zipcodes based on sum of the parking times
            
            df5 = df.groupby(['postalCode'])['soakTime'].sum().reset_index()
            df55 = df5.sort_values(by = ['soakTime'], ascending=False).reset_index()
            df_top_zips = df55['postalCode'].head(10)            
            print(df_top_zips)
          
            # =============================================================================
            # Find the ranking of the target area
            print(df55)
            
            target_area_time = df5['soakTime'].sum()                    
            
            appending_row = {'postalCode': 99999, 'soakTime' :target_area_time}
            df6 = df7.append(appending_row, ignore_index = True)
            df66 = df6.sort_values(by = ['soakTime'], ascending=False).reset_index()
            target_area_rank = df66[df66['postalCode']== 99999].index.values
            # print(df6)
            print(df66)
            print(f'target_area_time:{target_area_time}')
            print(f'target_area_rank: {target_area_rank}')
            percentage = (target_area_rank/len(df66)) * 100
            # print(f'type a = {type(percentage)}')
            round(percentage[0],0)
            print(f'Your area is among top {int(percentage[0])}% zipcodes with highest parking time.')
            # target_area_rank = df66.index
            # =============================================================================  
            
            zoom, center = zoom_center(list(df["longitude"]),list(df["latitude"]))
            
            print(center)
            lat_min = center['lat'] - boundaries
            lat_max = center['lat'] + boundaries
            lat_value = center['lat']
            print(lat_min, lat_max, lat_value)
            
            
            lon_min = center['lon'] - boundaries
            lon_max = center['lon'] + boundaries
            lon_value = center['lon']
            print(lon_min, lon_max, lon_value)
        
                    
            fig1 = px.scatter_mapbox(df,
                                    title= f'<b>{area}</b>- {period}',
                                    lat="latitude", 
                                    lon="longitude", 
                                    hover_name="postalCode", 
                                    hover_data=["soakTime", 'model', 'year'],
                                    color='model',
                                    opacity=0.5,
                                    # color_discrete_sequence=["fuchsia"], 
                                    zoom = zoom, 
                                    height = 600,
                                    center = center
                                    )
            fig1.update_layout(mapbox_style="open-street-map")
            fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            fig1.update_layout(title= f'<b>{area}</b>- {period}')
            
            trace4 = fig1.data[0]
            
            # =============================================================================
            
            df2 = df.groupby(['Date','Hour'])['soakTime'].count().reset_index()
            df22 = df2.groupby('Hour')['soakTime'].mean().reset_index()
            
            trace3 = go.Bar(x=df22['Hour'],
                            y= [int(i) for i in df22['soakTime']], 
                            name= 'Parked Vehicles Count',
                            xaxis="x3",
                            yaxis="y3")
            check = [int(i) for i in df22['soakTime']]
            # =============================================================================
            
            df3 = df.groupby('Hour')['soakTime'].mean().reset_index()
            df3['meanTime(min)'] = [int(i) for i in df3['soakTime']]
            
            trace2 = go.Bar(x=df3['Hour'], 
                            y=df3['meanTime(min)'], 
                            name= 'Mean Stop Time (min)',
                            xaxis="x2",
                            yaxis="y2",
                            )
            
            # =============================================================================
            
            df4 = df.groupby(['Date','Hour'])['soakTime'].sum().reset_index()
            df44 = df4.groupby('Hour')['soakTime'].mean().reset_index()
            
            trace1 = go.Bar(x=df44['Hour'], 
                            y=[int(i/60) for i in df44['soakTime']], 
                            name= 'Cumulative Stop Time (hr)',
                            marker={'color': 'black'},
                            xaxis="x",
                            yaxis="y"
                            )
            # trace1.update_traces(marker_color='rgb(158,202,225)')
    
            # =============================================================================
            
            data = [trace1, trace2, trace3, trace4]
            
            layout = go.Layout(
                xaxis=dict(
                    title="<b>Hour</b>",
                    domain=[0, 0.32],
    
                    # dtick = [x for x in range(12)],
                    # tickangle = 45
                    # titlefont=dict(
                    #     color="#1f77b4"
                    # ),
                    # tickfont=dict(
                    #     color="#1f77b4")
                ),
                xaxis2=dict(
                    # title="<b>Hour</b>",
                    domain=[0, 0.32],
                    anchor="y2",
                    showticklabels=False
                ),
                xaxis3=dict(
                    # title="<b>Hour</b>",
                    domain=[0, 0.32],
                    anchor="y3",
                    showticklabels=False
                ),                        
                xaxis4=dict(
                    domain=[0.33, 1],
                    anchor="y4"
                ),
                yaxis=dict(
                    title= '<b>Sum of Parking<br>Time (hr)</b>',
                    domain=[0, 0.31]
                ),
                yaxis2=dict(
                    title="<b>Average Parking<br>Time (min)</b>",
                    domain=[0.35, 0.66]
                ),
                yaxis3=dict(
                    title='<b>Count</b>',
                    domain=[0.7, 1]
                ),
                yaxis4=dict(
                    domain=[0, 1],
                    anchor="x4"
                )
            )
            
            fig = go.Figure(data=data, layout=layout)
            
            
            
            fig.update_layout(mapbox_style = "open-street-map",
                              mapbox = dict(
                                  accesstoken = mapbox_access_token,
                                  domain = {'x': [0.33, 1], 'y': [0.0, 1]},
                                  bearing = 0,
                                  center = center,
            #                      pitch=0,
                                  zoom = zoom - 1.5
                                  ),)
            
            fig.update_layout(height = 600, 
                              showlegend=False,
                              title_text = f'<b>{areatitle[area]}</b> ({period} 2019)',
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              # fillcolor= 'rgba(136, 204, 238, 0.5)'
                              )
            fig['layout']['xaxis']['mirror'] = True
            fig['layout']['xaxis']['linecolor'] = 'black'
    
            fig['layout']['xaxis2']['mirror'] = True
            fig['layout']['xaxis2']['linecolor'] = 'black'                               # Why does not work?
    
            fig['layout']['xaxis3']['mirror'] = True
            fig['layout']['xaxis3']['linecolor'] = 'black'
    
            fig['layout']['xaxis4']['mirror'] = True
            fig['layout']['xaxis4']['linecolor'] = 'black'
            
            fig['layout']['yaxis']['mirror'] = True
            fig['layout']['yaxis']['linecolor'] = 'black'
            
            fig['layout']['yaxis2']['mirror'] = True
            fig['layout']['yaxis2']['linecolor'] = 'black'
            
            fig['layout']['yaxis3']['mirror'] = True
            fig['layout']['yaxis3']['linecolor'] = 'black'
            
            fig['layout']['yaxis4']['mirror'] = True
            fig['layout']['yaxis4']['linecolor'] = 'black'
    
            # =============================================================================
        
            error_message = f'Top 10 zipcodes (out of {len(df55)}): {list(df_top_zips)}'
            if 'eval_button' in clicked_button:
                error_message = f'Your area is among top {int(percentage[0])}% zipcodes with highest parking time.'
                fig.update_layout(
                    mapbox=dict(
                        accesstoken=mapbox_access_token,
                        layers=layers,
                        bearing=0,
                        center=dict(
                            lat=lat_value,
                            lon=lon_value, 
                        ),
                        pitch=0,
                        zoom=zoom - 1.5,
                        style='outdoors')
                   )
                    
    return display, fig, error_message, lat_min, lat_max, lat_value, lon_min,\
           lon_max,\
           lon_value,\
           outputexport
                #, evalmessage 


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export_button", "n_clicks"),
    
    [State('dfout', 'data'),
    State('cutoff', 'value'),
    State('period', 'value'),
    State('area', 'value'),
    State('model', 'value')],
    
    prevent_initial_call=True,
    )

def func(n_clicks, data, cutoff, period, area, model):
    out= pd.DataFrame(data)
    # fmodel = model[0]
    return dcc.send_data_frame(out.to_csv, f"{area}_{period}_{cutoff}min.csv")

@app.callback(
    Output('evaluation_pan','style'),
    [Input('graph','style')]
    )

def evaluation_pan_show(graph_style):
    if graph_style['display'] == 'none':
        display = {'display': 'none'}
    else:
        display = {'display': 'block'}
    return display


if __name__ == '__main__':
    app.run_server(debug = False)
    
