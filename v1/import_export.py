import numpy as np
import pandas as pd
from modules.model_process import octave_thirdoctave, freq_c

#-----------------------------------IMPORT-----------------------------------#

def import_material(material_name):
    '''
    Input a string with the desired material and returns a dict with its 
    density, Young's modulus, loss factor and Poisson's modulus.
    The data will be imported from an excel file called 'tabla_materiales.xlsx'
    '''
    xl = pd.ExcelFile('tabla_materiales.xlsx')
    df = xl.parse('Materiales')   
    a = df['Material'] == material_name
        
    for i in range(len(a)):
        
        if a[i]==True:
    
            datos = df.iloc[i]
            
            parameters = {'name':material_name,
              'density': datos.loc['Densidad'],
              'young': datos.loc['Módulo de Young'],
              'poisson': datos.loc['Módulo Poisson'],
              'loss factor': datos.loc['Factor de pérdidas']}
    
    return parameters

#-----------------------------------EXPORT-----------------------------------#

def export_to_excel(filename, lx, ly, thickness, band_type, Material):
     
    freqs = octave_thirdoctave(band_type)
    fc = freq_c(lx, ly, thickness, Material.young, Material.poisson, Material.density)
    df_input = pd.DataFrame({
            'Material': [Material.name],
            'Lx [m]': [lx],
            'Ly [m]': [ly],
            'Espesor [m]': [thickness],
            'fc': [np.round(fc,2)]
            })
    

    df_output = pd.DataFrame({
            'Frecuencia [Hz]': freqs,
            'Modelo de Cremer': np.round(Material.cremer(lx,ly,thickness),2),
            'Modelo de Sharp': np.round(Material.sharp(lx,ly,thickness),2),
            'ISO 12354-1': np.round(Material.iso12354(lx,ly,thickness),2),
            'Modelo de Davy': np.round(Material.davy(lx,ly,thickness),2)}) 
    
    
    df_output = df_output.transpose()       

    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    
    df_input.to_excel(writer, sheet_name='Export', startrow=1, startcol=1, index=False)
    df_output.to_excel(writer, sheet_name='Export', startrow=4, header=False)
    
    workbook = writer.book
    worksheet = writer.sheets['Export']

    ### Changing column width
    worksheet.set_column(0, 0, 20)
    worksheet.set_column(1, 31, 7)

    ### Plotting the 4 arrays
    chart = workbook.add_chart({'type': 'line'})
    

    chart.add_series({
        'name':         '=Export!$A$6',
        'categories':   '=Export!$B$5:$AF$5',
        'values':       '=Export!$B$6:$AF$6'})

    chart.add_series({
        'name':         '=Export!$A$7',
        'categories':   '=Export!$B$5:$AF$5',
        'values':       '=Export!$B$7:$AF$7'})

    chart.add_series({
        'name':         '=Export!$A$8',
        'categories':   '=Export!$B$5:$AF$5',
        'values':       '=Export!$B$8:$AF$8'})

    chart.add_series({
        'name':         '=Export!$A$9',
        'categories':   '=Export!$B$5:$AF$5',
        'values':       '=Export!$B$9:$AF$9'})
    
    # Configure the chart axes.
    chart.set_x_axis({'name': 'Frequency [Hz]', 'position_axis': 'on_tick'})
    chart.set_y_axis({'name': 'R [dB]', 'major_gridlines': {'visible': False}})
    # Insert the chart into the worksheet.
    worksheet.insert_chart('C11', chart, {'x_scale': 3, 'y_scale': 1})
    
    writer.save()
    
