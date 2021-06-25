# -*- coding: utf-8 -*-
"""

Created on Tue Jun 22 19:05:41 2021
@author: m-askariyeh

"""

### Importing packages
import pandas as pd
import requests
import plotly.express as px
from flask import Flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


# =============================================================================
# 
# =============================================================================

symbolname = 'aapl'
start_year = 2010
end_year = 2020
years = list(range(2000,2022))
years_dic = {years[i]: years[i] for i in range(len(years))}



# path = '/Users/mhaskariyeh/Google Drive (1)/python_projects/TDI/Milestone Project'


#import os
# Setting up the server
server = Flask('my_app')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
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
             width="270",
             height="50"
             ),
         
         html.H6("TDI Milestone Project developed by:"),
         
         html.Div([
             html.A("Mohammad H. Askariyeh",
                    href="https://www.linkedin.com/in/mhaskariyeh/",
                    target="_blank",
                    style={'fontSize':20,
                           'font-style': 'italic',
                           'color': 'blue',
                           'justify-content': 'left',
                           'verticalAlign': 'middle',
                           'textAlign': 'left',
                           'alignItems': 'left',
                           }
                    ),
             
             ],
             style={'width':'100%',
                    'height': '50px',
                    'verticalAlign': 'middle'}
                 ),
         
         
         html.Br(),
         
         

         html.Div([ ## First Drop Down
                   html.H6("Stock Symbol:",
                           style={'width': "150px",
                                  'display': 'inline-block'}),
                   dcc.Input(id='stock_symbol',
                             placeholder="Stock Symbol",
                             value= 'MSFT',
                             type='text',
                             size = "20",
                             autoComplete = "off",
                             style={#'width': "30px",
                                    'display': 'inline-block',
                                    'lineHeight': '20px',
                                    'height':'30px' },)],
                  style={'verticalAlign': 'middle'},
                  
                  #className = "textbox"
                  ),                   

         
         
         html.Div([ ## start year Drop Down
                   html.H6("Start Year:",
                           style={'width': "150px",
                        'display': 'inline-block',
                        'lineHeight': '30px',
                        'height':'30px',
                        'verticalAlign': 'middle'}),
                   dcc.Dropdown(id = "startyear",
                                multi=False,
                                placeholder="Start Year",
                                options=[{'label': i, 'value': i} for i in years],
                                value= 2000,
                                style={'width': "215px",
                                       'display': 'inline-block',
                                       'lineHeight': '30px',
                                       'height':'30px',
                                       'verticalAlign': 'middle'})
                     ],
                  style={'verticalAlign': 'middle',
                         }
                  #className = "dropdown"
                  ),
         
                  
         html.Div([ ## Third Drop Down
                   html.H6("End Year:",
                           style={'width': "150px",
                                  'display': 'inline-block'}
                           ),
                   dcc.Dropdown(id = "endyear",
                                multi=False,
                                placeholder="End Year",
                                options=[{'label': i, 'value': i} for i in years],
                                value= 2021,
                                style={'width': "215px",
                                       'display': 'inline-block',
                                       'lineHeight': '30px',
                                       'height':'30px',
                                       'verticalAlign': 'middle'})
                   ],
                  #className = "dropdown"
                  style={'verticalAlign': 'middle',
                         'margin': '1px',
                         }
                  ),
         html.Button(children=['Submit'],
                     id='submit_button',
                     n_clicks=0,
                     className = 'profile_button'),
         
		], className = "split left"),
	html.Div([ ##  Right panel
           
			html.H2("Stock Open and Close Prices Time Series using Plotly Dash"),

			dcc.Graph(id = 'timeseries'),
            
            html.Div(id='errormessage')
            ], 
          className = "split right")
    ])

@app.callback(
    [Output('timeseries', 'style'),
     Output('timeseries', 'figure'),
     Output('errormessage', 'children')],
    [Input('submit_button', "n_clicks")],
    
    [State('stock_symbol', 'value'),
    State('startyear', 'value'),
    State('endyear', 'value')]
    )

    # Output('errormessage', 'value'),
    
def update_graph(submit_button, stock_symbol, startyear, endyear):
    error_message = 'Ready!'
    display =  {'display': 'none'}
    fig = {}
    if submit_button > 0:
        if startyear > endyear:
            error_message = 'Start year bigger than end year'
        else:
       
            api_key = 'api_key' #### Update the api_key
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&outputsize=full&apikey={api_key}'
            r = requests.get(url)
            data = r.json()
            try:
                display = {'display': 'flex', 'justify-content': 'center'}
                df = pd.DataFrame(data['Time Series (Daily)']).T
                df = df.reset_index()
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                df["Date"] = pd.to_datetime(df["Date"])
                df['Open'] = df['Open'].astype(float)
                df['Close'] = df['Close'].astype(float)
                
                df['Volume'] = df['Volume'].astype(int)
                df['Year'] = df["Date"].dt.year
                
                mask = (df['Year'] >= startyear) & (df['Year'] <= endyear)
                df = df.loc[mask]
                
                #  Y Label
                fig = px.line(df,
                              x= 'Date',
                              y= ['Open', 'Close'],
                              title= f'<b>{stock_symbol.upper()} Daily Stock Data</b> ({startyear} to {endyear})',
                              hover_data={'Date': '|%B %d, %Y'},
                              labels = dict(Date = '<b>Year</b>',
                                            value = '<b>Price</b> (USD)',
                                            variable = '<b>Price</b>'),
                              color_discrete_sequence= ['blue', '#10B498'])
                fig.update_layout(title_x = 0.5,
                              title_y = 0.85,
                              title_font_size = 25,
                              paper_bgcolor= '#F4F6F8',
                              font_color= '#060662')
                error_message = 'Done!'
            except:
                error_message = 'Maximun data request reached. Please wait one minute and try again.'
                fig = {}
                display =  {'display': 'none'}
                    
 
    return display, fig, error_message 


if __name__ == '__main__':
    app.run_server(debug=True)
