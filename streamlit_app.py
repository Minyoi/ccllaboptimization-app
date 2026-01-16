import streamlit as st

import pandas as pd
import numpy as np
from pandasql import sqldf
import altair as alt
import pandas as pd
import altair as alt
import plotly.express as px
alt.renderers.enable("html")

import streamlit as st
from streamlit_gsheets import GSheetsConnection

url = "https://docs.google.com/spreadsheets/d/1W7KvSPMLNHFKYC9fh_l3t9cbt3_GsSxe4fU4vXzhUB8/edit?usp=sharing"

conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read(spreadsheet=url)
# st.dataframe(df)

df['SAMPLE ID'] = df['SAMPLE ID'].replace('NULL', np.nan)
df.dropna(subset=['SAMPLE ID'], inplace=True)

df['PATIENT ID'] = df['PATIENT ID'].replace('NULL', np.nan)
df.dropna(subset=['PATIENT ID'], inplace=True)

df['TEST RESULT'] = df['TEST RESULT'].replace('NULL', np.nan)
df.dropna(subset=['TEST RESULT'], inplace=True)

df['TEST CODE'] = df['TEST CODE'].replace('NULL', np.nan)
df.dropna(subset=['TEST CODE'], inplace=True)

df['ACCESSION NUMBER'] = df['ACCESSION NUMBER'].replace('NULL', np.nan)
df.dropna(subset=['ACCESSION NUMBER'], inplace=True)


df['sampleid_testcode'] = df['PATIENT ID'].astype(str) + '-'+ df['TEST CODE'].astype(str)+'-'+ df['ACCESSION NUMBER'].astype(str)
df['rec_date'] = pd.to_datetime(df['RECEIVE DATE'], format='%m/%d/%Y')
df['val_date'] = pd.to_datetime(df['VALIDATION DATE'],format='%m/%d/%Y')
df['turnaround_time'] = (df['val_date'] - df['rec_date']).dt.days

df['month_processed'] = df['rec_date'].dt.to_period('M')
df['year_processed'] = df['rec_date'].dt.to_period('Y')
a = ['AFBST','XPUT','CULTB','XPRIF']
new_df = df[df['TEST CODE'].isin(a)]
pivoted = new_df.pivot(index=["sampleid_testcode","RECEIVE DATE", "month_processed","year_processed", "rec_date","val_date","turnaround_time"], columns="TEST CODE", values="TEST RESULT")
#pivoted = pd.pivot_table(new_df,values='TEST RESULT',index=["sampleid_testcode","RECEIVE DATE", "month_processed","year_processed", "rec_date","val_date","turnaround_time"],columns='TEST CODE')
new_pivoted = pivoted.reset_index()


new_pivoted['smear_result'] = np.where(new_pivoted['AFBST']=='NAFB',0,np.where(new_pivoted['AFBST']=='2AFB',1,np.where(new_pivoted['AFBST']=='MTBND',0,np.where(new_pivoted['AFBST']=='1AFB',1,np.where(new_pivoted['AFBST']=='3AFB',1,np.where(new_pivoted['AFBST']=='ZTEST',2,np.where(new_pivoted['AFBST']=='SAFB',1,np.nan)))))))
new_pivoted['smear_lowgrade'] = np.where(new_pivoted['AFBST']=='NAFB',0,np.where(new_pivoted['AFBST']=='2AFB',0,np.where(new_pivoted['AFBST']=='MTBND',0,np.where(new_pivoted['AFBST']=='1AFB',1,np.where(new_pivoted['AFBST']=='3AFB',0,np.where(new_pivoted['AFBST']=='SAFB',1,np.nan))))))
new_pivoted['xpert_result'] = np.where(new_pivoted['XPUT']=='MTBND',0,np.where(new_pivoted['XPUT']=='ERR',2,np.where(new_pivoted['XPUT']=='MTBVL',1,np.where(new_pivoted['XPUT']=='Mycobacteria Tuberculosis Trace Detected',1,np.where(new_pivoted['XPUT']=='MTBL',1,np.where(new_pivoted['XPUT']=='MTBH',1,np.where(new_pivoted['XPUT']=='ZTEST',6,np.where(new_pivoted['XPUT']=='Mycobacteria  Tuberculosis Trace Detected',1,np.where(new_pivoted['XPUT']=='Mycobacteria Tuberculosis Detected Trace',1,np.where(new_pivoted['XPUT']=='MTBM',1,np.where(new_pivoted['XPUT']=='INSUF',8,np.where(new_pivoted['XPUT']=='ND',0,np.where(new_pivoted['XPUT']=='Mycobacterium Tuberculosis Trace detected',1,np.nan)))))))))))))
new_pivoted['culture_result'] = np.where(new_pivoted['CULTB']=='TBCP',1,np.where(new_pivoted['CULTB']=='TBCN',0,np.where(new_pivoted['CULTB']=='TBCC',4,np.where(new_pivoted['CULTB']=='INSUF',5,np.where(new_pivoted['CULTB']=='ZTEST',3,np.where(new_pivoted['CULTB']=='TF1',2,np.where(new_pivoted['CULTB']=='MOTT',1,np.where(new_pivoted['CULTB']=='NEG',0,np.where(new_pivoted['CULTB']=='TF2',2,np.where(new_pivoted['CULTB']=='MTBC',1,np.nan))))))))))
new_pivoted['rif_result'] = np.where(new_pivoted['XPRIF']=='RRND',0,np.where(new_pivoted['XPRIF']=='ZTEST',2,np.where(new_pivoted['XPRIF']=='NA',4,np.where(new_pivoted['XPRIF']=='INSUF',5,np.where(new_pivoted['XPRIF']=='RRIN',6,np.where(new_pivoted['XPRIF']=='RRDT',1,np.nan))))))
new_pivoted['rif_det'] = np.where(new_pivoted['XPRIF']=='RRND',0,np.where(new_pivoted['XPRIF']=='INSUF',0,np.where(new_pivoted['XPRIF']=='RRIN',0,np.where(new_pivoted['XPRIF']=='RRTD',1,np.nan))))
new_pivoted['xpos_rfdet'] = np.where((new_pivoted['xpert_result'] == 1)&(new_pivoted['rif_det']==1),1,np.where((new_pivoted['xpert_result'] == 1)&(new_pivoted['rif_det']==0),0,np.nan))
new_pivoted['smear_rec_rate_den'] = np.where((new_pivoted['AFBST']=='1AFB')&((new_pivoted['CULTB'] =='TBCP') | (new_pivoted['CULTB'] =='TBCN')),1,np.where((new_pivoted['AFBST']=='2AFB')&((new_pivoted['CULTB']=='TBCP') | (new_pivoted['CULTB'] =='TBCN')),1,np.where((new_pivoted['AFBST']=='3AFB')&((new_pivoted['CULTB']=='TBCP')| (new_pivoted['CULTB'] =='TBCN')),1,np.where((new_pivoted['AFBST']=='NAFB')&((new_pivoted['CULTB']=='TBCP')| (new_pivoted['CULTB'] =='TBCN')),1,0))))
new_pivoted['smear_rec_rate_num'] = np.where((new_pivoted['AFBST']=='1AFB')&(new_pivoted['CULTB'] =='TBCP'),1,np.where((new_pivoted['AFBST']=='2AFB')&(new_pivoted['CULTB']=='TBCP'),1,np.where((new_pivoted['AFBST']=='3AFB')&(new_pivoted['CULTB']=='TBCP'),1,np.where((new_pivoted['AFBST']=='NAFB')&(new_pivoted['CULTB']=='TBCP'),1,0))))
        

culture_df = new_pivoted[['month_processed','year_processed' ,'culture_result','turnaround_time']]
xpert_df = new_pivoted[['month_processed','year_processed', 'xpert_result','turnaround_time']]
smear_df = new_pivoted[['month_processed','year_processed', 'smear_result','turnaround_time']]
smearlg_df = new_pivoted[['month_processed','year_processed', 'smear_lowgrade','turnaround_time']]
rif_df = new_pivoted[['month_processed','year_processed', 'rif_result','turnaround_time']]
xpos_rifdet_df = new_pivoted[['month_processed','year_processed', 'xpos_rfdet','turnaround_time']]
smear_recovery_df = new_pivoted[['month_processed','year_processed', 'smear_rec_rate_den', 'smear_rec_rate_num']]
         


(culture_df
 .groupby(['month_processed', 'culture_result'])
 .size()
 .unstack()
 .plot.bar()
)


st.set_page_config(
    page_title="Lab Optimization Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="collapsed")



df1 = culture_df['month_processed'].value_counts().rename_axis('unique_values').reset_index(name='counts')
df1=pd.crosstab(index=culture_df['month_processed'], columns=culture_df['culture_result'])
df1 = df1.rename(columns = {0.0:'Negative', 1.0:'Positive',2.0:'TF', 3.0:'Z_test',4.0:'Contaminated',5.0:'Insufficient'})
df1_1 = df1.copy()
df1_1['total_samples'] = df1[['Contaminated','Positive','Negative','TF','Z_test']].sum(axis=1) 
df1_1['p_contaminated'] = df1_1['Contaminated']/df1_1['total_samples'] * 100
df1_1['p_negative'] = df1_1['Negative']/df1_1['total_samples'] * 100
df1_1['p_positive'] = df1_1['Positive']/df1_1['total_samples'] * 100
df1_2 = df1_1.copy()
df1_2.drop(['Contaminated', 'Negative', 'Positive','TF', 'Z_test','total_samples'], axis='columns', inplace=True)
df1_3 = culture_df.copy()
culture_tat = pd.crosstab(index=df1_3['month_processed'],
                           columns=df1_3['culture_result'],
                           values=df1_3['turnaround_time'],
                           aggfunc='mean')
culture_tat = culture_tat.rename(columns = {0.0:'Negative', 1.0:'Positive',2.0:'TF', 3.0:'Z_test',4.0:'Contaminated',5.0:'Insufficient'})
        

df2=pd.crosstab(index=xpert_df['month_processed'], columns=xpert_df['xpert_result'])

df2 = df2.rename(columns = {0.0:'MTB_Not_detected', 1.0:'MTB_Detected',2.0:'Error', 6.0:'Z_test',8.0:'Insufficient'})
df2_1 = df2.copy()
df2_1['total_samples'] = df2[['Insufficient','Error','MTB_Detected','MTB_Not_detected','Z_test']].sum(axis=1) 
df2_1['p_mtbdetected'] = df2_1['MTB_Detected']/df2_1['total_samples'] * 100
df2_1['p_error'] = df2_1['Error']/df2_1['total_samples'] * 100
df2_2 = df2_1.copy()
df2_2.drop(['Insufficient', 'Error', 'MTB_Detected', 'MTB_Not_detected','Z_test','total_samples'], axis='columns', inplace=True)
df2_3 = xpert_df.copy()
xpert_tat = pd.crosstab(index=df2_3['month_processed'],
                           columns=df2_3['xpert_result'],
                           values=df2_3['turnaround_time'],
                           aggfunc='mean')
xpert_tat = xpert_tat.rename(columns = {0.0:'MTB_Not_detected', 1.0:'MTB_Detected',2.0:'Error', 6.0:'Z_test',8.0:'Insufficient'})
df5=pd.crosstab(index=xpos_rifdet_df['month_processed'], columns=xpos_rifdet_df['xpos_rfdet'])
df5 = df5.rename(columns = {0.0:'MTB Det RIF Negative', 1.0:'MTB_Det RIF Positive'})
df5_1 = df5.copy()



def add_column_if_not_exists(df, column_name):
  """Adds a column to a DataFrame if it doesn't exist and fills it with zeros.

    Args:
        df: The Pandas DataFrame.
        column_name: The name of the column to add.

    Returns:
        The modified DataFrame.
  """
  if column_name not in df.columns:
    df[column_name] = 0
  return df



df5_1 = add_column_if_not_exists(df5_1, 'MTB_Det RIF Positive')
df5_1['total_samples'] = df5_1[['MTB Det RIF Negative','MTB_Det RIF Positive']].sum(axis=1) 
df5_1['p_positive'] = df5_1['MTB_Det RIF Positive']/df5_1['total_samples'] * 100
        



df3=pd.crosstab(index=smear_df['month_processed'], columns=smear_df['smear_result'])
df3 = df3.rename(columns = {0.0:'TB_Not_detected', 1.0:'TB_Detected',2.0:'Z_test'})
df3_1 = df3.copy()
df3_1['total_samples'] = df3[['TB_Not_detected','TB_Detected','Z_test']].sum(axis=1) 
df3_1['p_positive'] = df3_1['TB_Detected']/df3_1['total_samples'] * 100
df3_1['p_negative'] = df3_1['TB_Not_detected']/df3_1['total_samples'] * 100
df3_2 = df3_1.copy()
df3_2.drop(['TB_Not_detected', 'TB_Detected', 'Z_test','total_samples'], axis='columns', inplace=True)

df2_3 = smear_df.copy()
smear_tat = pd.crosstab(index=df2_3['month_processed'],
                           columns=df2_3['smear_result'],
                           values=df2_3['turnaround_time'],
                           aggfunc='mean')
smear_tat = smear_tat.rename(columns = {0.0:'TB_Not_detected', 1.0:'TB_Detected',2.0:'Z_test'})


#Low grade processing
df4=pd.crosstab(index=smear_df['month_processed'], columns=smearlg_df['smear_lowgrade'])
df4 = df4.rename(columns = {0.0:'Not_Lowgrade', 1.0:'Low_Grade'})
df4_1 = df4.copy()
df4_1['total_smear'] = df4[['Not_Lowgrade','Low_Grade']].sum(axis=1) 
df4_1['p_lowgrade'] = df4_1['Low_Grade']/df4_1['total_smear'] * 100
        
df_smear_join = df4_1.join(df3_1, how='inner')
df6 = smear_recovery_df.groupby('month_processed')[['smear_rec_rate_den','smear_rec_rate_num']].sum()
df6['recovery_rate'] = (df6['smear_rec_rate_num'] / df6['smear_rec_rate_den']) * 100
df6.drop(['smear_rec_rate_den', 'smear_rec_rate_num'], axis='columns', inplace=True)              

with st.sidebar:
    st.title('🏂 CCL Lab Optimization dashboard')

    quarter_list = list(culture_df.month_processed.unique())[::-1]
    quarter_list.sort()
    # file = st.file_uploader("Please upload the lab data extract below")
    st.write("Results processed from the following months")
    st.dataframe(quarter_list)

    # st.dataframe(culture_df)
    

col = st.columns((3, 3, 3), gap='medium')

with col[0]:
    st.markdown('#### Culture')
        
    st.write('Total Culture Samples processed')
    df1_1 = df1_1.reset_index() 
    df1_1['month_processed'] = df1_1['month_processed'].astype(str)
    st.bar_chart(df1_1,x="month_processed",y="total_samples")

    st.write("Culture Quality Indicators")
    df1_2 = df1_2.reset_index()
    #st.write(df1_2)
    df1_2['month_processed'] = df1_2['month_processed'].astype(str)
    st.line_chart(df1_2, x='month_processed', y_label='percentage')
    st.write("Culture Turnaround time (days)")
    culture_tats = culture_tat.reset_index()
    culture_tats['month'] = culture_tats['month_processed'].astype(str)
    culture_tats.drop('month_processed', axis=1)

    a = alt.Chart(culture_tats).mark_area(opacity=0.6, color="#308b8b").encode(x='month', y='Positive')
    b = alt.Chart(culture_tats).mark_area(opacity=0.6, color="#b14a8e").encode(x='month', y='Negative')
    c = alt.Chart(culture_tats).mark_area(opacity=0.6, color="#1751cc").encode(x='month', y='TF')
    d = alt.Chart(culture_tats).mark_area(opacity=0.6, color="#151d2e").encode(x='month', y='Z_test')
    e = alt.Chart(culture_tats).mark_area(opacity=0.6, color='#beaed4').encode(x='month', y='Contaminated')
    f = alt.layer(a,b,c,d,e)
    st.altair_chart(f, use_container_width=True)

    # New culture turnaround time data transformed
    data_long = pd.melt(culture_tats, id_vars=['month'], value_vars=['Negative','Positive','TF','Z_test','Contaminated'], var_name='results', value_name='days')
    
    # Create an Altair chart
    chart = alt.Chart(data_long).mark_line().encode(
        x='month',
        y='days:Q',  
        color='results:N'  # Use the category field for color encoding
    ).properties(
        title="Turn around time (Culture)"
    )
    
    line = alt.Chart(pd.DataFrame({'y': [60]})).mark_rule().encode(y='y')
    st.altair_chart(chart+line, use_container_width=True)
    

    # st.dataframe(culture_tat)
    st.write("Culture Results")
    st.dataframe(df1)

    with st.expander('Key', expanded=True):
        st.write('''
            - Culture: Lab data processed upto ''' + str(max(quarter_list)) + '''
            ''')   

with col[1]:
    st.markdown('#### Xpert')
    
    st.write('Total Xpert Samples processed')
    df2_1 = df2_1.reset_index() 
    df2_1['month_processed'] = df2_1['month_processed'].astype(str)
    st.bar_chart(df2_1, x="month_processed", y="total_samples")

    st.write("Xpert Quality Indicators")
    df2_2 = df2_2.reset_index()
    # st.write(df2_2)
    df2_2['month_processed'] = df2_2['month_processed'].astype(str)
    st.line_chart(df2_2, x='month_processed', y_label='percentage')
    st.write("Xpert Turnaround time (days)")
    xpert_tats = xpert_tat.reset_index()
    xpert_tats['month'] = xpert_tats['month_processed'].astype(str)
    xpert_tats.drop('month_processed', axis=1)

    a = alt.Chart(xpert_tats).mark_area(opacity=0.6, color="#308b8b").encode(x='month', y='MTB_Not_detected')
    b = alt.Chart(xpert_tats).mark_area(opacity=0.6, color="#2c1270").encode(x='month', y='MTB_Detected')
    c = alt.Chart(xpert_tats).mark_area(opacity=0.6, color="#c93919").encode(x='month', y='Error')
    d = alt.Chart(xpert_tats).mark_area(opacity=0.6, color="#aad021").encode(x='month', y='Insufficient')
    e = alt.layer(a, b,c,d)
    st.altair_chart(e, use_container_width=True)

    # New xpert turnaround time data transformed
    xdata_long = pd.melt(xpert_tats, id_vars=['month'], value_vars=['MTB_Not_detected','MTB_Detected','Error','Insufficient'], var_name='results', value_name='days')
    
    # Create an Altair chart
    chart = alt.Chart(xdata_long).mark_line().encode(
        x='month',
        y='days:Q',  
        color='results:N'  # Use the category field for color encoding
    ).properties(
        title="Turn around time (Xpert)"
    )
    
    line1 = alt.Chart(pd.DataFrame({'y': [2]})).mark_rule().encode(y='y')
    st.altair_chart(chart+line1, use_container_width=True)


    # st.dataframe(xpert_tat)
    st.write("Xpert results")
    st.dataframe(df2)

    with st.expander('Key', expanded=True):
        st.write('''
            - Xpert: Lab data processed upto ''' + str(max(quarter_list)) + '''
            ''')   

with col[2]:
    st.markdown('#### Smear')
    st.write('Total Smear Samples processed')
    df3_1 = df3_1.reset_index() 
    df3_1['month_processed'] = df3_1['month_processed'].astype(str)
    st.bar_chart(df3_1,x='month_processed', y="total_samples")

    st.write("Smear Quality Indicators")
    df3_2 = df3_2.reset_index()
    # st.write(df3_2)
    df3_2['month_processed'] = df3_2['month_processed'].astype(str)
    df_smear_join = df_smear_join.reset_index()
        
    df_smear_join.drop(['Not_Lowgrade', 'Low_Grade','total_samples','Z_test', 'TB_Detected','TB_Not_detected','total_smear','p_negative'], axis='columns', inplace=True)
    df_smear_join['month_processed'] = df_smear_join['month_processed'].astype(str)
    st.line_chart(df_smear_join, x='month_processed', y_label='percentage')
    
    st.write("Smear Turnaround time (days)")
    smear_tats = smear_tat.reset_index()
    smear_tats['month'] = smear_tats['month_processed'].astype(str)
    smear_tats.drop('month_processed', axis=1)

    a = alt.Chart(smear_tats).mark_area(opacity=0.6, color="#308b8b").encode(x='month', y='TB_Not_detected')
    b = alt.Chart(smear_tats).mark_area(opacity=0.6, color="#080bd3").encode(x='month', y='TB_Detected')

    c = alt.layer(a, b)

    st.altair_chart(c, use_container_width=True)

    # New smear turnaround time data transformed
    sdata_long = pd.melt(smear_tats, id_vars=['month'], value_vars=['TB_Not_detected','TB_Detected'], var_name='results', value_name='days')
    
    # Create an Altair chart
    chart = alt.Chart(sdata_long).mark_line().encode(
        x='month',
        y='days:Q',  
        color='results:N'  # Use the category field for color encoding
    ).properties(
        title="Turn around time (Smear)"
    )
    
    line2 = alt.Chart(pd.DataFrame({'y': [2]})).mark_rule().encode(y='y')
    st.altair_chart(chart+line2, use_container_width=True)

    # Display line graph for recovery rate
    st.write("Recovery rate")
    df6 = df6.reset_index() 
    df6['month_processed'] = df6['month_processed'].astype(str)
    st.line_chart(df6, x='month_processed')


    # st.dataframe(smear_tat)
    st.write("Smear results")
    st.dataframe(df3)
    # st.write(df3_1)


    with st.expander('Key', expanded=True):
        st.write('''
            - Smear: Lab data processed upto ''' + str(max(quarter_list)) + '''

            ''')   




