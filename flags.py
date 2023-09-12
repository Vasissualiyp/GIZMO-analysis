import re

#Enabling of different parts of the code
SinglePlotMode=False
plotting=True
eternal_plotting=False
custom_loader=False
sph_plotter=False
#For 2D plots
colorbarlims=False
custom_center=True 
#For 1D plots
wraparound = True
#For double plotting
double_plot=False
InitialPlotting=True
#Data analysis
Time_Dependent=False

#loader/upscaler options
RestrictedLoad=True
debugging=True


def get_flags_array():
    variables = []
    pattern = r'\b[A-Za-z_]+\b\s*=\s*(True|False)'
    with open(__file__, 'r') as f:
        for line in f:
            matches = re.findall(pattern, line)
            if matches:
                variable_name = line.split('=')[0].strip()
                if matches[0] == 'True':
                    variables.append(variable_name)
    return variables
