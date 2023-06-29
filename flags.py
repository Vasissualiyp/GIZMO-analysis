import re

#Enabling of different parts of the code
SinglePlotMode=False
plotting=True
custom_loader=True
sph_plotter=True
#For 2D plots
colorbarlims=False
custom_center=False 
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
