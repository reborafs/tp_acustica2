import matplotlib.pyplot as plt
from import_export import import_material, export_to_excel
from Material_class import Material
from model_process import octave_thirdoctave


# INPUT
material_name = 'Acero'
material = Material(import_material(material_name))
thickness = 0.01
lx = 8
ly = 3

# CALCULUS
band_type = 'third'
freqs = octave_thirdoctave(band_type)

R_cremer = material.cremer(lx,ly,thickness, band_type)
R_sharp = material.sharp(lx,ly,thickness)
R_iso = material.iso12354(lx,ly,thickness)
R_davy = material.davy(lx,ly,thickness)

# Export to excel
export_to_excel(lx,ly,thickness, band_type, material)

# PLOTTING
plt.figure()
plt.title('Single leaf wall insulation')
plt.semilogx(freqs, R_cremer)
plt.semilogx(freqs, R_sharp)
plt.semilogx(freqs, R_iso)
plt.semilogx(freqs, R_davy)
plt.xlabel('Frequency [Hz]')
plt.ylabel('R [dB]')
plt.xticks((20,100,1000,10000,20000),(20,100,'1k','10k','20k'))
plt.grid(which='both')

plt.legend(('Cremer', 'Sharp', 'ISO 12354-1', 'Davy'))