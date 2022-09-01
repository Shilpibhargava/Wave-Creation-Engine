import base64
import datetime
import io

import dash
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output , State, dash_table  # pip install dash (version 2.0.0 or higher)
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[1]").appName("wave_tool").getOrCreate()
from pyspark.sql.functions import col,row_number
from pyspark.sql.types import StructType
import pyspark.sql.functions as f
df= pd.DataFrame()
import pyspark.pandas as ps

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,suppress_callback_exceptions=True)

app.css.config.serve_locally = False

app.css.append_css({"external_url": "./assets/xyz.css"})

# for 3840
d1=[[1,55,55,0],[2,8,508,500],[3,8,508,500],[4,8,508,500],[5,8,508,500],[6,8,508,500],[7,8,508,500],[8,8,428,420],[9,8,508,500],[10,8,508,500],[11,8,508,500],
    [12,8,508,500],[13,10,390,380],[14,43,293,250]]

# for 3842
d2=[[1,58,58,0],[2,8,500,492],[3,8,509,501],[4,8,792,784],[5,8,792,784],[6,8,792,784],[7,8,792,784],[8,8,446,438],[9,8,792,784],[10,8,720,712],[11,8,792,784],
    [12,8,792,784],[13,10,576,566],[14,43,528,485],[15,8,386,378],[16,56,56,0]]


aisles=pd.DataFrame(d1,columns=['Aisle_No',	'Pallet Locations',	'Practically Usable Locations','C+B locations'])



#file_name='BAS_20220720013522 7.22.2022 B.xlsx'
#data_raw=pd.read_excel('/Users/Z003CNF/Library/CloudStorage/OneDrive-TargetCorporation/UDC_PTS/'+file_name,sheet_name= 'DistroData' , header =5, converters={'Distro':str,'DPCI':str})
#data_raw.columns=data_raw.columns.str.replace(' ', '').str.replace('\n','')
#aisles=pd.read_excel('/Users/Z003CNF/Library/CloudStorage/OneDrive-TargetCorporation/UDC_PTS/'+file_name,sheet_name= 'Aisles')
#print(aisles)

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Wave Creation Engine", style={'text-align': 'center'}),

    dcc.Upload(id='upload-data',
        children=html.Div([
            'Upload the drop file by clicking here',
            #html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True),
    html.Div(id='output-summary'),
    html.Div(id='output-datatable'),
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            data_raw = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            data_raw = pd.read_excel(io.BytesIO(decoded),sheet_name= 'DistroData' , header =5, converters={'Distro':str,'DPCI':str})
            data_raw.columns = data_raw.columns.str.replace(' ', '').str.replace('\n', '')
            #aisles = pd.read_excel(io.BytesIO(decoded),sheet_name='Aisles')

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        html.Button("Download Waves", id="btn_csv"),
        dcc.Download(id="download-dataframe-csv"),

        html.P("Select a Store Stock Zone"),

        dcc.Dropdown(id="slct_SSZ",
                     options=[{'label': x, 'value': x}
                              for x in sorted(data_raw['SSZ'].unique())],
                     multi=False,
                     value='SSZ',
                     style={'width': "40%"}
                     ),
        html.Br(),

        html.P("Select a Aisle for wave size"),

        dcc.Dropdown(id="slct_Aisle",
                     options=[{'label':x, 'value':x}
                                for x in sorted(aisles['Aisle_No'].unique())],
                     multi=False,
                     value='Aisle',
                     style={'width': "40%"}
                    ),
        html.Br(),

        html.P("Is Unit-Sortable Y/N ?"),

        dcc.Dropdown(id="slct_fgt",
                     options=[{'label':x, 'value':x}
                                for x in sorted(data_raw['UnitSortable'].unique())],
                     multi=False,
                     value='Unit-Sortable',
                     style={'width': "40%"}
                     ),
        html.Br(),

        html.Button(id="submit-button", children="Create Waves"),

        html.Br(),
        html.Hr(),

        dash_table.DataTable(
            data=data_raw.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in data_raw.columns],
            page_size=5
        ),
        dcc.Store(id='stored-data', data=data_raw.to_dict('records')),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        #html.Div('Raw Content'),
        #html.Pre(contents[0:200] + '...', style={
           # 'whiteSpace': 'pre-wrap',
           # 'wordBreak': 'break-all'
        #})
    ])

@app.callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output('output-summary', 'children'),
              Input('submit-button', 'n_clicks'),
              State('stored-data', 'data'),
              State('slct_SSZ', 'value'),
              State('slct_Aisle', 'value'),
              State('slct_fgt', 'value') )

def build_waves(n,data_raw,user_input_ssz,user_input_aisle,unit_sortable):
        if n is None:
            return dash.no_update
        else :
            print(type(data_raw))
            ## convert list to pandas df
            data_raw1=pd.DataFrame(data_raw)
            print(unit_sortable)
            print(type(data_raw1))
            data_raw2=data_raw1.loc[(data_raw1['UnitSortable']==unit_sortable)]
            print(type(data_raw2))
            df_wave_dpci_dims = spark.createDataFrame(data_raw2)

            ###---------------------------------------------- Calculations at Distro grain ---------------------------------------------------###

            # Data prep at distro - grain for loc - calcs
            df_wave_dpci_dims1 = df_wave_dpci_dims.select('Distro', 'DPCI', 'SSZ',
                                                          col('TotalNumberofFullVCPs').alias('vcp'),
                                                          col('TotalNumberofLooseSSP').alias('ssp'),
                                                          col('TotalNumberofPallets').alias('pallet'),
                                                          col('SSPsinVCP').alias('ssp_q'),
                                                          'VCPHeight', 'VCPWidth', 'VCPLength',
                                                          col('Aisle').alias('wh_aisle'))

            data = df_wave_dpci_dims1.withColumn("RatioSSPsinBox", col('ssp') / col('ssp_q'))
            data1 = data.select(data['*'], f.sort_array(f.array('VCPHeight', 'VCPWidth', 'VCPLength')).alias("arr"))
            data2 = data1.select(data1['*'], data1.arr[0].alias('Ht'), data1.arr[1].alias('Wt'),
                                 data1.arr[2].alias('Lt')).drop('VCPHeight', 'VCPWidth', 'VCPLength', 'arr')

            # Location's estimation for VCP boxes - basic assumption boxes are charged one behind another on longer side (length)
            # Location dimensions (15.5 X 92.5 X 15)

            # One VCP box takes how many locations width wise ## f.lit(1) ## for historical validations
            data2 = data2.withColumn("Loc_wt", f.ceil(
                col('Wt') / 15.5))  # 15.5/col('Wt')# for future 15.5/shorter side of box (width) # long==length, # shorter==width # shortest==height

            # How many VCP can be Put one behind another on length
            data2 = data2.withColumn("Loc_lt", f.floor((92.5 / col('Lt'))))

            # How many VCP Can be stacked one on top? 1st stack (first stack can be just 2 high)
            data2 = data2.withColumn("Loc_ht1", f.when(col('Ht') <= 5, 2).otherwise(0))

            # Subsequesnt Stacks (2nd stack onwards can be 3 or more depending on height of box) # flaps tolerance still missing # 5 inch is benchmark to stack # levels a,b,c,d categorization also not there
            data2 = data2.withColumn("Loc_ht2", f.when(col('Ht') <= 5, f.ceil((15 / col('Ht')))).otherwise(0))

            # Total VCP boxes that can fit in 'Loc_lt' location (Level B, C)
            data2 = data2.withColumn("Loc_tot", f.when(col('Loc_ht1') > 0,
                                                       col('Loc_ht1') + (col('Loc_lt') - 1) * col('Loc_ht2')).otherwise(
                col('Loc_lt')))

            # Can all vcp's in this distro be fit in AG's locations ? (this is actually space remaining and these many more boxes could have come)
            data2 = data2.withColumn("Loc_vcp_rmng",
                                     col('Loc_tot') - col('vcp') - f.when(col('RatioSSPsinBox') > 0.5, 1).otherwise(0))

            # If Loc_vcp_rmng is negative ,then extra locations are needed for those remaining vcp's other wise zero additional location, ##default 1 location width wise--col('Loc_wt')
            data2 = data2.withColumn("Loc_extra", f.when(col('Loc_vcp_rmng') <= 0, col('Loc_wt') * f.ceil(
                f.abs(col('Loc_vcp_rmng') / col('Loc_tot')))).otherwise(0) + col('Loc_wt'))

            # is there Space Remaining for tray to fit ?
            data2 = data2.withColumn("Loc_tray", f.when((((col('Loc_extra') * col('Loc_tot')) - col('vcp') - f.when(
                col('RatioSSPsinBox') > 0.5, 1).otherwise(0)) * col('Lt')) > 7.7, 1).otherwise(0))

            # If No space to Keep Tray then one additional location needed
            data2 = data2.withColumn("Loc_extra_tray", f.when(col('Loc_tray') == 0, 1).otherwise(0))

            # JUST LOOSE SSP's FLAG
            data2 = data2.withColumn("Loose_ssp_flag", f.when((col('vcp') == 0) & (col('ssp') > 0), 1).otherwise(0))

            # Total Locations
            data2 = data2.withColumn("Total_locs", f.when(col('Loose_ssp_flag') == 0,
                                                          col('Loc_extra') + col('Loc_extra_tray')).otherwise(f.when(col('RatioSSPsinBox') > 0, 1).otherwise(0)))
            data211 = data2.select('Distro', 'SSZ', 'DPCI', 'Total_locs').toPandas()

            data222 = pd.merge(data_raw2, data211, how='left')

            ###---------------------------------------------------------- Calculations at DPCI grain--------------------------------------####

            #### Data prep at dpci grain for loc-calcs

            # filter non-pallet distro's
            df_wave_dpci_dims1 = df_wave_dpci_dims1.filter(df_wave_dpci_dims1.pallet == 0)

            data_dpci_raw = df_wave_dpci_dims1.groupBy('SSZ', 'DPCI').agg(f.sum('vcp').alias('vcp1'),
                                                                          f.sum('ssp').alias('ssp'),
                                                                          f.sum('pallet').alias('pallet'),
                                                                          f.max('ssp_q').alias('ssp_q'),
                                                                          f.max('VCPHeight').alias('vend_cspk_ht_q'),
                                                                          f.max('VCPWidth').alias('vend_cspk_wth_q'),
                                                                          f.max('VCPLength').alias('vend_cspk_lgth_q'))

            data = data_dpci_raw
            data = data.withColumn("RatioSSPsinBox1", col('ssp') / col('ssp_q'))
            data = data.withColumn("vcp", col('vcp1') + f.floor(col('RatioSSPsinBox1')))
            data = data.withColumn("RatioSSPsinBox", col('RatioSSPsinBox1') - f.floor(col('RatioSSPsinBox1')))

            data11 = data.select(data['*'],
                                 f.sort_array(f.array('vend_cspk_ht_q', 'vend_cspk_wth_q', 'vend_cspk_lgth_q')).alias(
                                     "arr"))
            data21 = data11.select(data11['*'], data11.arr[0].alias('Ht'), data11.arr[1].alias('Wt'),
                                   data11.arr[2].alias('Lt')).drop('vend_cspk_ht_q', 'vend_cspk_wth_q',
                                                                   'vend_cspk_lgth_q', 'arr', 'vcp1', 'RatioSSPsinBox1')

            # Location's estimation for VCP boxes basic assumption boxes are charged one behind another on longer side (length)
            # Location dimensions (15.5 X 92.5 X 15)

            # One VCP box takes how many locations width wise
            data21 = data21.withColumn("Loc_wt", f.ceil(
                col('Wt') / 15.5))  # 15.5/col('Wt')# for future 15.5/shorter side of box (width) # long==length, # shorter==width # shortest==height

            # How many VCP can be Put one behind another on length
            data21 = data21.withColumn("Loc_lt", f.floor((92.5 / col('Lt'))))

            # How many VCP Can be stacked one on top? 1st stack (first stack can be just 2 high)
            data21 = data21.withColumn("Loc_ht1", f.when(col('Ht') <= 5, 2).otherwise(0))

            # Subsequesnt Stacks (2nd stack onwards can be 3 or more depending on height of box) # flaps tolerance still missing # 5 inch is benchmark to stack # levels a,b,c,d categorization also not there
            data21 = data21.withColumn("Loc_ht2", f.when(col('Ht') <= 5, f.ceil((15 / col('Ht')))).otherwise(0))

            # Total VCP boxes that can fit in 'Loc_lt' location (Level B, C)
            data21 = data21.withColumn("Loc_tot", f.when(col('Loc_ht1') > 0, col('Loc_ht1') + (col('Loc_lt') - 1) * col(
                'Loc_ht2')).otherwise(col('Loc_lt')))

            # Can all vcp's in this distro be fit in AG's locations ? (this is actually space remaining and these many more boxes could have come)
            data21 = data21.withColumn("Loc_vcp_rmng",
                                       col('Loc_tot') - col('vcp') - f.when(col('RatioSSPsinBox') > 0.5, 1).otherwise(
                                           0))

            # If Loc_vcp_rmng is negative ,then extra locations are needed for those remaining vcp's other wise zero additional location, ##default 1 location width wise--col('Loc_wt')
            data21 = data21.withColumn("Loc_extra", f.when(col('Loc_vcp_rmng') <= 0, col('Loc_wt') * f.ceil(
                f.abs(col('Loc_vcp_rmng') / col('Loc_tot')))).otherwise(0) + col('Loc_wt'))

            # is there Space Remaining for tray to fit ?
            data21 = data21.withColumn("Loc_tray", f.when((((col('Loc_extra') * col('Loc_tot')) - col('vcp') - f.when(
                col('RatioSSPsinBox') > 0.5, 1).otherwise(0)) * col('Lt')) > 7.7, 1).otherwise(0))

            # If No space to Keep Tray then one additional location needed
            data21 = data21.withColumn("Loc_extra_tray", f.when(col('Loc_tray') == 0, 1).otherwise(0))

            # JUST LOOSE SSP's FLAG
            data21 = data21.withColumn("Loose_ssp_flag", f.when((col('vcp') == 0) & (col('ssp') > 0), 1).otherwise(0))

            # Total Locations
            data21 = data21.withColumn("Total_locs", f.when(col('Loose_ssp_flag') == 0,
                                                            col('Loc_extra') + col('Loc_extra_tray')).otherwise(
                f.when(col('RatioSSPsinBox') > 0, 1).otherwise(0)))

            # Bring DPCI grain dataframe to distro level dataframe to be able to consume well, also the total_pts locations will
            # correspond to just one distro,

            data211 = data21.select('SSZ', 'DPCI', 'Total_locs').toPandas()

            data_raw11 = data_raw2.loc[(data_raw2['TotalNumberofPallets'] == 0)]
            data31 = pd.merge(data_raw11, data211, how='left')

            data31['rank'] = data31.groupby(['DPCI'])['Distro'].rank(method='first', ascending=False)

            data31.loc[data31['rank'] > 1, ['Total_locs']] = 0
            # data31.loc[data31['rank'] > 1, ['pallet']] = 0

            #####---------------------------------------------Making Waves from distro-level location calcs (exclude pallets)------------------------####

            ## User input values of Aisle & SSZ
            ## Everything SSZ9 goes to A14, all SSZ 2 goes to A13.

            #user_input_ssz = 13  ## 2,9
            #user_input_aisle = 2  ## 13,14

            #### Make a choice of using the distro-grain v/s dpci-grain

            if user_input_ssz == 2:
                distro_locs = data31
            else:
                distro_locs = data222

            ## wave size corresponding to cartons+bins locations
            Wave_size_ctns = aisles.loc[aisles['Aisle_No'] == user_input_aisle, 'C+B locations'].tolist()[0]

            ##filter out non-pallet distro's and specifc SSZ
            df1 = distro_locs.loc[(distro_locs['TotalNumberofPallets'] == 0) & (distro_locs['SSZ'] == user_input_ssz)]

            ##Sort the data by asec wh_aisle, for clubbing nearby wharehosuing locations together in a wave.
            distro_locs = df1.sort_values(by=['Aisle'])

            ## Allocate WaveID's
            distro_locs['wave_cum_locations'] = distro_locs['Total_locs'].cumsum()
            distro_locs['wave_id'] = 'W' + np.ceil(distro_locs['wave_cum_locations'].div(Wave_size_ctns)).astype(int).astype(str)
            distro_locs['wave_no'] = np.ceil(distro_locs['wave_cum_locations'].div(Wave_size_ctns)).astype(int)

            max_wave_cartons = distro_locs['wave_no'].max()

            #####------------------------------------------------ Pallet Locations Calc-----------------------------------------############

            Wave_size_pallets = aisles.loc[aisles['Aisle_No'] == user_input_aisle, 'Pallet Locations'].tolist()[0]
            pract_usable_locs = aisles.loc[aisles['Aisle_No'] == user_input_aisle, 'Practically Usable Locations'].tolist()[0]

            pallet_distro = data_raw2.loc[(data_raw2['TotalNumberofPallets'] > 0) & (data_raw2['SSZ'] == user_input_ssz)]

            pallet_dpci = pallet_distro.groupby('DPCI').agg(
                pallets=pd.NamedAgg(column="TotalNumberofPallets", aggfunc="sum"),
                vcp=pd.NamedAgg(column="TotalNumberofFullVCPs", aggfunc="sum"),
                VCPHt=pd.NamedAgg(column="VCPHeight", aggfunc="mean")).reset_index(level=0)

            # Sort all the dpci by total no. of VCP boxes, pallets with larger no. of boxes should be sent to pallet locations
            # rather than pallets with smaller quantity of boxes.
            # in winters add a height ocnstraint too

            pallet_dpci.sort_values(by=['vcp', 'VCPHt'], ascending=False, inplace=True)

            # one pallet location assumed to hold max of 3 pallets.
            pallet_dpci["pallet_loc"] = np.ceil(pallet_dpci["pallets"] / 3)

            ## total pallets in each wave
            pallet_dpci['wave_cum_locations'] = pallet_dpci['pallet_loc'].cumsum()

            ## allocate Wave ID's
            pallet_dpci['wave_id'] = 'W' + np.ceil(pallet_dpci['wave_cum_locations'].div(Wave_size_pallets)).astype(int).astype(str)
            pallet_dpci['wave_no'] = np.ceil(pallet_dpci['wave_cum_locations'].div(Wave_size_pallets)).astype(int)

            max_wave_pallets = pallet_dpci['wave_no'].max()

            ##Join back to raw data to get distro_level data
            pallet_waves = pd.merge(pallet_distro, pallet_dpci, how='left')

            # .drop(['pallets','vcp','VCPHt','pallet_loc','wave_cum_locations'], axis=1)

            pallet_distro = pallet_waves.loc[pallet_waves['wave_no'] <= max_wave_cartons]

            ##################### Make additional pallet waves/general waves based on trailing pallets#####################

            plts_rmng = pd.DataFrame(columns=['wave_no'])

            if max_wave_pallets > max_wave_cartons:
                plts_rmng = pallet_waves.loc[pallet_waves['wave_no'] > max_wave_cartons]

                if plts_rmng['TotalNumberofPallets'].sum() > 100:
                    plts_rmng.loc[:, 'wave_id'] = 'Pallet_wave'
                else:
                    ## Assuming if there are lesser pallets then they can be charged to carton locations
                    plts_locs = pd.merge(plts_rmng, data222, how='inner')
                    ##Sort the data by asec wh_aisle, for clubbing nearby wharehosuing locations together in a wave.
                    plts_rmng = plts_locs.sort_values(by=['Aisle'])
                    ## Allocate WaveID's
                    plts_rmng['wave_cum_locations'] = distro_locs['Total_locs'].cumsum()
                    plts_rmng['wave_id'] = 'W_P' + np.ceil(
                        distro_locs['wave_cum_locations'].div(Wave_size_ctns)).astype(int).astype(str)

            distro_locs = distro_locs.drop(['wave_no', 'wave_cum_locations'], axis=1)
            pallet_distro = pallet_distro.drop(['wave_no'], axis=1)
            plts_rmng = plts_rmng.drop(['wave_no'], axis=1)

            ### -------------------------------------------------------------------Concat and Summary---------------------------------------####

            df = pd.concat([distro_locs, pallet_distro, plts_rmng], ignore_index=True, sort=False)  # plts_rmng

            summary = df.groupby(by=['wave_id']).agg(
                ssz=pd.NamedAgg(column="SSZ", aggfunc="mean"),
                Total_plts=pd.NamedAgg(column="TotalNumberofPallets", aggfunc="sum"),
                Total_vcp=pd.NamedAgg(column="TotalNumberofFullVCPs", aggfunc="sum"),
                Total_ssp=pd.NamedAgg(column="TotalNumberofSSPsintheDistro", aggfunc="sum"),
                Total_distro=pd.NamedAgg(column="Distro", aggfunc="count"),
                Total_dpci=pd.NamedAgg(column="DPCI", aggfunc="nunique"),
                Total_CandB_locs=pd.NamedAgg(column="Total_locs", aggfunc="sum")).reset_index(level=0)

            summary['Aisle_utilz'] = (summary['Total_CandB_locs'] + Wave_size_pallets) / (pract_usable_locs + 5)
            summary['Aisle_utilz'] = summary['Aisle_utilz'].astype(float).map("{:.2%}".format)

            summary.sort_values(by=['Aisle_utilz'])

            data = summary.to_dict('rows')
            print(type(data))
            columns = [{"name": i, "id": i, } for i in (summary.columns)]
            return dash_table.DataTable(data=data, columns=columns)

            #return html.Div([ dash_table.DataTable(data=summary.to_dict('records'),columns=[{'name': i, 'id': i} for i in summary.columns],page_size=15)])


@app.callback(Output("download-dataframe-csv", "data"),Input("btn_csv", "n_clicks"))

def dwnld_data(n,df):
    if n is None:
        return dash.no_update
    else:
        return dcc.send_data_frame(df.to_csv, "mydf.csv")

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=False, port=8050)
    #app.run_server(debug=True)
    
#,port=8060