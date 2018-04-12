# Basic Imports
import numpy as np
# Plotting Imports
import matplotlib
import matplotlib.pyplot as plt
import corner
# Chempy Imports
from Chempy.parameter import ModelParameters
from Chempy.sfr import SFR
from Chempy.wrapper import Chempy
from Chempy.solar_abundance import solar_abundances
from Chempy.data_to_test import plot_processes
# Widget Imports
import ipywidgets as widgets

#==========================================================
# WIDGET GENERATION
#==========================================================

#---------------------------------------
# MODEL INPUTS WIDGETS
#---------------------------------------

#..........................
# Basic Inputs
#..........................
def create_TimeInput():
    time_range_label = widgets.Label(value='Time Range (Gyr):')
    time_range = widgets.FloatRangeSlider(
                                          value=[0.0, 13.5],
                                          min=0.0,
                                          max=13.5,
                                          step=0.1,
                                          disabled=False,
                                          continuous_update=True,
                                          orientation='horizontal',
                                          readout=True,
                                          readout_format='.1f',
                                          layout=widgets.Layout(width='99%')
                                          )
    time_step_label = widgets.Label(value='Time Step (Gyr):')
    time_step = widgets.FloatSlider(
                                    value=0.5,
                                    min=0.025,
                                    max=0.5,
                                    step=0.005,
                                    disabled=False,
                                    continuous_update=True,
                                    orientation='horizontal',
                                    readout=True,
                                    readout_format='.3f',
                                    layout=widgets.Layout(width='99%')
                                    )
    return(time_range_label,time_range,time_step_label,time_step)

#..........................
# Star Formation
#..........................
def create_SFR():
    SFR_dict = {'Just & Jahreiss 2010':0,'Gamma PDF':1,'From File':2, 'Doubly Peaked':3}
    SFR_fun_label = widgets.Label(value='SFR Function:')
    SFR_fun = widgets.Dropdown(
                               options=SFR_dict,
                               value=1,
                               layout=widgets.Layout(width='99%')
                               )
    return(SFR_fun_label,SFR_fun)

def create_SFR_gamma():
    SFR_gamma_latex = widgets.Label(
                                    value="$$\\psi(t) \\propto \\frac{[(t-t_{0})/C]^{(a-1)}\ e^{(t_{0}-t)/C}}{C\\gamma(a)}$$",
                    layout=widgets.Layout(height='150%')
                                    )
    SFR_gamma_a_label = widgets.Label(value="$$\\text{SFR Beginning:}\ t_{0}$$")
    SFR_gamma_a = widgets.FloatSlider(
                                      value=0.0,
                                      min=0.0,
                                      max=10.00,
                                      step=0.01,
                                      readout_format='.2f',
                                      layout=widgets.Layout(width='99%')
                                      )
    SFR_gamma_b_label = widgets.Label(value="$$\\text{SFR Scale:}\ C$$")
    SFR_gamma_b = widgets.FloatSlider(
                                      value=3.5,
                                      min=0.0,
                                      max=10.0,
                                      step=0.01,
                                      readout_format='.2f',
                                      layout=widgets.Layout(width='99%')
                                      )
    SFR_gamma_c_label = widgets.Label(value="$$\\text{Shape Parameter}:\ a = 2$$")
    return(SFR_gamma_latex,
           SFR_gamma_a_label,SFR_gamma_a,
           SFR_gamma_b_label,SFR_gamma_b,
           SFR_gamma_c_label)

def create_SFR_prescribed():
    SFH_dict = {'Aquarius (DDO 210)':'ddo210','IC 1613':'ic1613','Leo A':'leo_a'}
    SFR_prescribed_a_label = widgets.Label(value='Observed SFH:')
    SFR_prescribed_a = widgets.Dropdown(
                                        options = SFH_dict,
                                        value = 'ddo210',
                                        layout=widgets.Layout(width='99%')
                                        )
    return(SFR_prescribed_a_label,SFR_prescribed_a)


def create_SFR_doublypeaked():
    SFR_doublypeaked_latex = widgets.Label(
                                           value="$$\\psi(t) \\propto \\frac{e^{[(t_{0,1}-t)/C]^{2}/2}}{\\int_{t_{start}}^{t_{end}}e^{[(t_{0,1}-t)/C]^{2}/2}dt} + A\\frac{e^{(t_{0,2}-t)/C}}{\\int_{t_{start}}^{t_{end}}e^{(t_{0,2}-t)/C}dt}$$",
                                    layout=widgets.Layout(height='150%')
                                    )
    SFR_doublypeaked_a_label = widgets.Label(value="$$\\text{Peak 1 Beginning:}\ t_{0,1}$$")
    SFR_doublypeaked_a = widgets.FloatSlider(
                                             value=0.8,
                                             min=0.0,
                                             max=10.00,
                                             step=0.01,
                                             readout_format='.2f',
                                             layout=widgets.Layout(width='99%')
                                             )
    SFR_doublypeaked_b_label = widgets.Label(value="$$\\text{Peak 1 Scale:}\ C_{1}$$")
    SFR_doublypeaked_b = widgets.FloatSlider(
                                             value=0.8,
                                             min=0.0,
                                             max=10.00,
                                             step=0.01,
                                             readout_format='.2f',
                                             layout=widgets.Layout(width='99%')
                                             )
    SFR_doublypeaked_c_label = widgets.Label(value="$$\\text{Peak 2 Beginning:}\ t_{0,2}$$")
    SFR_doublypeaked_c = widgets.FloatSlider(
                                             value=2.0,
                                             min=0.0,
                                             max=10.00,
                                             step=0.01,
                                             readout_format='.2f',
                                             layout=widgets.Layout(width='99%')
                                             )
    SFR_doublypeaked_d_label = widgets.Label(value="$$\\text{Peak 2 Decay Parameter:}\ C_{2}$$")
    SFR_doublypeaked_d = widgets.FloatSlider(
                                             value=3.5,
                                             min=0.0,
                                             max=10.00,
                                             step=0.01,
                                             readout_format='.2f',
                                             layout=widgets.Layout(width='99%')
                                             )
    SFR_doublypeaked_e_label = widgets.Label(value="$$\\text{Peak Ratio:}\ A$$")
    SFR_doublypeaked_e = widgets.FloatSlider(
                                             value=0.8,
                                             min=0.0,
                                             max=10.00,
                                             step=0.01,
                                             readout_format='.2f',
                                             layout=widgets.Layout(width='99%')
                                             )
    return(SFR_doublypeaked_latex,
           SFR_doublypeaked_a_label,SFR_doublypeaked_a,
           SFR_doublypeaked_b_label,SFR_doublypeaked_b,
           SFR_doublypeaked_c_label,SFR_doublypeaked_c,
           SFR_doublypeaked_d_label,SFR_doublypeaked_d,
           SFR_doublypeaked_e_label,SFR_doublypeaked_e
           )

def create_SFR_modelA():
    SFR_modelA_latex = widgets.Label(
                                     value="$$\\psi(t) \\propto \\frac{t+t_{0}}{(t^2 + t_1^2)^2}$$",
                                     layout=widgets.Layout(height='150%')
                                    )
    SFR_modelA_a_label = widgets.Label(value="$$t_{0}$$")
    SFR_modelA_a = widgets.FloatSlider(
                                      value=5.6,
                                      min=0.0,
                                      max=10.00,
                                      step=0.01,
                                      readout_format='.2f',
                                      layout=widgets.Layout(width='99%')
                                      )
    SFR_modelA_b_label = widgets.Label(value="$$t_1$$")
    SFR_modelA_b = widgets.FloatSlider(
                                      value=8.2,
                                      min=0.0,
                                      max=10.0,
                                      step=0.01,
                                      readout_format='.2f',
                                      layout=widgets.Layout(width='99%')
                                     )
    return(SFR_modelA_latex,
           SFR_modelA_a_label,SFR_modelA_a,
           SFR_modelA_b_label,SFR_modelA_b)

#..........................
# Inflow / Outflow
#..........................


def create_infall():
    infall_dict = {'Primordial':0,'Solar':1,'Simple':2,'Alpha':3}
    infall_label = widgets.Label(value='Infall Type:')
    infall = widgets.Dropdown(
                              options=infall_dict,
                              value=0,
                              layout=widgets.Layout(width='99%')
                              )
    return(infall_label,infall)


def create_outflow():
    outflow_label = widgets.Label(value='Outflow Fraction:')
    outflow = widgets.FloatSlider(
                                  value=0.5,
                                  min=0.0,
                                  max=1.0,
                                  step=0.05,
                                  disabled=False,
                                  continuous_update=True,
                                  orientation='horizontal',
                                  readout=True,
                                  readout_format='.2f',
                                  layout=widgets.Layout(width='99%')
                                  )
    return(outflow_label,outflow)

#..........................
# Display Selection
#..........................
def create_ModelSelection():
    model_seleciton_label = widgets.Label(value='Run chemical evolution for:')
    model_select_1 = widgets.Checkbox(description='Model #1',value=True,indent=False,
                                      layout=widgets.Layout(width='99%'))
    model_select_2 = widgets.Checkbox(description='Model #2',value=False,indent=False,
                                      layout=widgets.Layout(width='99%'))
    model_select_3 = widgets.Checkbox(description='Model #3',value=False,indent=False,
                                      layout=widgets.Layout(width='99%'))
    model_selection = widgets.VBox(children= [model_select_1,model_select_2,model_select_3])
    return(model_seleciton_label,model_selection)

def create_ElementSelection(a):
    list_of_elements = a.elements_to_trace
    list_of_elements.remove('H')
    #search_widget = widgets.Text()
    options_dict = {element: widgets.Checkbox(description=element, value=False, indent=False, layout=widgets.Layout(width='9%')) for element in list_of_elements}
    options = [options_dict[element] for element in list_of_elements]
    
    options[5].value = True # Ca
    options[11].value = True # Fe
    options[17].value = True # Mg
    options[27].value = True # Si
    options[23].value = True # O
    
    options_widget = widgets.HBox(options,
                                  layout=widgets.Layout(width='99%',
                                                        display='inline-flex',flex_flow='row wrap')
                                  )
                                      
    #ElementSelection = widgets.VBox([search_widget, options_widget])
    ElementSelection = widgets.VBox([options_widget])
                                                        
    # Wire the search field to the checkboxes
    #def on_text_change(change):
    #    search_input = change['new']
    #    if search_input == '':
    #        # Reset search field
    #        new_options = [options_dict[element] for element in list_of_elements]
    #    else:
    #        # Filter by search field using difflib.
    #        close_matches = difflib.get_close_matches(search_input, list_of_elements, cutoff=0.0)
    #        new_options = [options_dict[element] for element in close_matches]
    #    options_widget.children = new_options
    #
    #search_widget.observe(on_text_change, names='value')
    return(ElementSelection)

#..........................
# Buttons
#..........................

def create_ResetButton():
    reset = widgets.Button(
                         description='Reset!',
                         disabled=False,
                         button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
                         tooltip='Click me to reset to defaults!'
                         )
    return(reset)

def create_RunButton():
    run = widgets.Button(
                         description='Generate!',
                         disabled=False,
                         button_style='success', # 'success', 'info', 'warning', 'danger' or ''
                         tooltip='Click me to run model!',
                         layout=widgets.Layout(width='99%')
                         )
    return(run)


def create_PreviewButton():
    preview = widgets.Button(
                             description='Preview SFR',
                             disabled=False,
                             button_style='', # 'success', 'info', 'warning', 'danger' or ''
                             tooltip='Click me to preview SFR',
                             layout=widgets.Layout(width='99%')
                             )
    return(preview)

def create_ReplotButton():
    #align_kw = dict(margin = '100px 100px 100px 100px')
    replot = widgets.Button(
                             description='Replot Elements',
                             disabled=False,
                             button_style='', # 'success', 'info', 'warning', 'danger' or ''
                             tooltip='Click me to replot w/ new element selection'
                             )
    return(replot)

#..........................
# Progress
#..........................

def create_ProgressBar(max):
    progress_bar = widgets.IntProgress(
                                       value=0,
                                       min=0,
                                       max=max,
                                       step=0.1,
                                       bar_style='info',
                                       orientation='horizontal',
                                       layout=widgets.Layout(width='99%')
                                       )
    return(progress_bar)

#---------------------------------------
# WIDGET UPDATES
#---------------------------------------

#==========================================================
# CHEMPY
#==========================================================

#---------------------------------------
# INITIALIZE
#---------------------------------------
def update_a(basic_inputs,model_inputs):
    a = ModelParameters()
    a.elements_to_trace = ['Al', 'Ar', 'B', 'Be', 'C', 'Ca', 'Cl', 'Co', 'Cr', 'Cu', 'F', 'Fe', 'Ga', 'Ge', 'H', 'He', 'K', 'Li', 'Mg', 'Mn', 'N', 'Na', 'Ne', 'Ni', 'O', 'P', 'S', 'Sc', 'Si', 'Ti', 'V', 'Zn']
    a.check_processes = True

    a.start = basic_inputs[0] #time_range.value[0]
    a.end = basic_inputs[1] #time_range.value[1]
    a.time_steps = np.floor((a.end - a.start) / basic_inputs[2]) + 1 #time_step.value
    a.basic_sfr_index = model_inputs[0] #SFR_fun_1.value
    a.basic_sfr_name = a.basic_sfr_name_list[model_inputs[0]] #SFR_fun_1.value
    a.name_of_file = 'input/SFR/' + model_inputs[1] + '.lcid.final.sfh' #SFR_prescribed_1a.value
    a.name_infall_index = model_inputs[2] #infall_1.value
    a.name_infall = a.name_infall_list[model_inputs[2]] #infall_1.value
    a.outflow_feedback_fraction = model_inputs[3] #outflow_1.value
    return(a)


def update_SFR(a,sfr_input):
    
    SFR_gamma_1a = sfr_input[0]
    SFR_gamma_1b = sfr_input[1]
    SFR_prescribed_1a = sfr_input[2]
    SFR_2peak_1a = sfr_input[3]
    SFR_2peak_1b = sfr_input[4]
    SFR_2peak_1c = sfr_input[5]
    SFR_2peak_1d = sfr_input[6]
    SFR_2peak_1e = sfr_input[7]
    SFR_modelA_1a = sfr_input[8]
    SFR_modelA_1b = sfr_input[9]
    
    sfr = SFR(a.start,a.end,a.time_steps)
    if a.basic_sfr_name == 'model_A':
        a.mass_factor = 1.
        a.S_0 = 45.07488
        a.t_0 = SFR_modelA_1a
        a.t_1 = SFR_modelA_1b
        getattr(sfr, a.basic_sfr_name)(S0 = a.S_0 * a.mass_factor,
                                       t0 = a.t_0,
                                       t1 = a.t_1)
    if a.basic_sfr_name == 'gamma_function':
        a.mass_factor = 1.
        a.S_0 = 1#45.07488
        a.a_parameter = 2
        a.sfr_beginning = SFR_gamma_1a
        a.sfr_scale = SFR_gamma_1b # 3.5 SFR peak in Gyr for a = 2
        getattr(sfr, a.basic_sfr_name)(S0 = a.S_0 * a.mass_factor,
                                       a_parameter = a.a_parameter,
                                       loc = a.sfr_beginning,
                                       scale = a.sfr_scale)
    if a.basic_sfr_name == 'prescribed':
        a.mass_factor = 1.
        getattr(sfr, a.basic_sfr_name)(mass_factor = a.mass_factor,
                                       name_of_file = a.name_of_file)
    if a.basic_sfr_name == 'doubly_peaked':
        a.mass_factor = 1.
        a.S_0 = 45.07488
        a.peak_ratio = SFR_2peak_1e
        a.sfr_decay = SFR_2peak_1d
        a.sfr_t0 = SFR_2peak_1c
        a.peak1t0 = SFR_2peak_1a
        a.peak1sigma = SFR_2peak_1b
        getattr(sfr, a.basic_sfr_name)(S0 = a.S_0 * a.mass_factor,
                                       peak_ratio = a.peak_ratio,
                                       decay = a.sfr_decay,
                                       t0 = a.sfr_t0,
                                       peak1t0 = a.peak1t0,
                                       peak1sigma = a.peak1sigma)
    return(sfr)

#---------------------------------------
# EVOLVE
#---------------------------------------

def Evolve(**kwargs):
    
    cube = []
    abundances = []
    models = []
    for i,key in enumerate(kwargs):
        a = kwargs[key]
        cube_temp, abundances_temp = Chempy(a)
        cube.append(cube_temp)
        abundances.append(abundances_temp)
        models.append(key)

    CUBE = {x:y for x,y in zip(models, cube)}
    ABUNDANCES = {x:y for x,y in zip(models, abundances)}
    return(CUBE,ABUNDANCES)

#---------------------------------------
# SAMPLE STARS
#---------------------------------------

def sample_stars_new(weight,selection,elements,errors,nsample):
    weight = np.cumsum(weight*selection)
    weight /= weight[-1]
    sample = np.random.random(nsample)
    sample = np.sort(sample)
    stars = np.zeros_like(weight)
    for i,item in enumerate(weight):
        if i == 0:
            count = len(sample[np.where(np.logical_and(sample>0.,sample<=item))])
            stars[i] = count
        else:
            count = len(sample[np.where(np.logical_and(sample>weight[i-1],sample<=item))])
            stars[i] = count
    
    abundances = np.zeros((len(elements),nsample))
    n = 0
    for i in range(len(weight)):
        if stars[i] != 0:
            for j in range(int(stars[i])):
                for k in range(len(elements)):
                    abundances[k][n] = elements[k][i]
                n += 1
    abundances = np.array(abundances)
    for i,element in enumerate(elements):
        perturbation = np.random.normal(0,errors[i],len(abundances[i]))
        abundances[i] += perturbation
    return abundances


def SampleStars_New(a,N,elements_to_sample,**kwargs):
    
    basic_solar = solar_abundances()
    getattr(basic_solar, a.solar_abundance_name)()
    
    # Red Clump Selection Criteria
    selection_raw = np.load("input/selection/red_clump_new.npy")
    time_selection_raw = np.load("input/selection/time_red_clump_new.npy")
    
    sample = np.interp(kwargs['Model #1']['time'], time_selection_raw[::-1], selection_raw)
    selection = np.interp(kwargs['Model #1']['time'], time_selection_raw[::-1], selection_raw)
    
    x = np.zeros((len(kwargs),len(elements_to_sample),N))
    keys = sorted(kwargs.keys())
    sampled_abundances = []
    
    for i,key in enumerate(sorted(kwargs)):
        elements = []
        errors = []
        for element in elements_to_sample:
            if element == 'Fe':
                elements.append(kwargs[key][element][1:])
                errors.append(float(basic_solar.table['error'][np.where(basic_solar.table['Symbol']==element)]))
            else:
                elements.append(kwargs[key][element][1:]-kwargs[key]['Fe'][1:])
                errors.append(float(basic_solar.table['error'][np.where(basic_solar.table['Symbol']==element)]))
        x[i] = sample_stars_new(kwargs[key]['weights'][1:],selection[1:],elements,errors,N)
        sampled_abundances.append({y:z for y,z in zip(elements_to_sample, x[i])})
    sampled_abundances = {y:z for y,z in zip(keys, sampled_abundances)}

    return(sampled_abundances)


#=======================================================================
# PLOTTING
#=======================================================================
def plotSFR(**kwargs):
    fig, ax = plt.subplots()
    
    color=iter(plt.cm.viridis(np.linspace(0,1,len(kwargs))))
    for key in sorted(kwargs):
        c=next(color)
        t = kwargs[key].t
        sfr = kwargs[key].sfr
        sfr /= np.trapz(sfr,t)
        ax.plot(t,sfr,color=c,linestyle='-',label='%s' % key)

    ax.set(xlabel='Time (Gyr)',
           ylabel=r'SFR (M$_*$/Gyr)',
           title='Star Formation History')
    ax.legend(loc=1)

    plot = ax.figure
    return(plot)

def plotMDF(element,legend_loc=2,**kwargs):#,data):
    fig, ax = plt.subplots()
    
    color=iter(plt.cm.viridis(np.linspace(0,1,len(kwargs))))
    keys = sorted(kwargs.keys())
    bins = np.linspace(-4,1,51)
    for i,key in enumerate(sorted(kwargs)):
        c=next(color)
        ax.hist(kwargs[key][element][1:],bins=bins,color=c,histtype='step', label = '%s' % keys[i],density=False)
    
    if element == 'Fe':
        x_label = '[%s/H]' % element
        element_label = '[%s_H]' % element
        element_data = '__%s_H_' % element
    else:
        x_label = '[%s/Fe]' % element
        element_label = '[%s_Fe]' % element
        element_data = '__%s_Fe_' % element
    ax.set(xlabel=x_label,
           ylabel='#',
           title='Metallicity Distribution Function')
    ax.legend(loc=legend_loc)

    plot = ax.figure
    return(plot)

def plotCorner(elements_to_plot,**kwargs):
    
    label = []
    for element in elements_to_plot:
        if element == 'Fe':
            label.append(r'[%s/H]' % element)
        else:
            label.append(r'[%s/Fe]' % element)
    
    color=iter(plt.cm.viridis(np.linspace(0,1,len(kwargs))))

    lines = []
    for i,key in enumerate(sorted(kwargs)):
        c = next(color)
        data = []
        lines.append(matplotlib.lines.Line2D([], [], color=c, label=key))
        for element in elements_to_plot:
            data.append(kwargs[key][element])
        data = np.ndarray.transpose(np.vstack(data))
        if i == 0:
            figure = corner.corner(data,labels=label,color=c)
        else:
            figure = corner.corner(data,color=c,fig=figure)
    figure.legend(handles=lines, loc=1,
                  bbox_to_anchor=(0.95, 0.95),fontsize=(12+2*len(elements_to_plot)))
    return(figure)

def plotYieldContributions(element,legend_loc=4,**kwargs):
    fig, ax = plt.subplots()
    plot_lines = []
    lines = []
    linestyle=iter(['-','--',':'])
    
    for i,key in enumerate(sorted(kwargs)):
        ls=next(linestyle)
        l1, = ax.plot(kwargs[key].time,np.cumsum(kwargs[key].sn2_cube[element] + kwargs[key].sn1a_cube[element] + kwargs[key].agb_cube[element]),
                      linestyle = ls, linewidth=5,color='gray')
        l2, = ax.plot(kwargs[key].time,np.cumsum(kwargs[key].sn2_cube[element]),
                linestyle = ls, linewidth=1,color='b')
        l3, = ax.plot(kwargs[key].time,np.cumsum(kwargs[key].sn1a_cube[element]),
                linestyle = ls, linewidth=1,color='g')
        l4, = ax.plot(kwargs[key].time,np.cumsum(kwargs[key].agb_cube[element]),
                linestyle = ls, linewidth=1,color='r')
        plot_lines.append([l1,l2,l3,l4])
        lines.append(matplotlib.lines.Line2D([], [], color='k',linestyle = ls, label=key))
    
    legend1 = plt.legend(plot_lines[0], ['Total','CC-SN','SN Ia','AGB'], loc=4)
    ax.legend(handles=lines, loc=4,bbox_to_anchor=(0.8, 0.0))
    plt.gca().add_artist(legend1)

    ax.set(xlabel='Time (Gyr)',
           ylabel='Mass Fraction Expelled',
           title='%s Yield Contributions' %element,
           xscale='log',
           yscale='log')
           
    plot = ax.figure
    return(plot)

def plot_processes(summary_pdf,name_string,sn2_cube,sn1a_cube,agb_cube,elements,cube1,number_of_models_overplotted):
    '''
        This is a plotting routine showing the different nucleosynthetic contributions to the individual elements.
        INPUT:
        summary_pdf = boolean, should a pdf be created?
        
        name_string = string to be added in the saved file name
        
        sn2_cube = the sn2_feeback class
        
        sn1a_cube = the sn1a_feeback class
        
        agb_cube = the agb feedback class
        
        elements = which elements should be plotted
        
        cube1 = the ISM mass fractions per element (A Chempy class containing the model evolution)
        
        number_of_models_overplotted = default is 1, if more the results will be saved and in the last iteration all former models will be plotted at once
        
        OUTPUT:
        
        A plotfile in the current directory
        '''
    
    probability = 0
    
    sn2 = []
    agb = []
    sn1a= []
    for i,item in enumerate(elements):
        if item == 'C+N':
            sn2_temp = np.sum(sn2_cube['C'])
            sn2_temp += np.sum(sn2_cube['N'])
            sn2.append(sn2_temp)
            sn1a_temp = np.sum(sn1a_cube['C'])
            sn1a_temp += np.sum(sn1a_cube['N'])
            sn1a.append(sn1a_temp)
            agb_temp = np.sum(agb_cube['C'])
            agb_temp += np.sum(agb_cube['N'])
            agb.append(agb_temp)
        else:
            sn2.append(np.sum(sn2_cube[item]))
            sn1a.append(np.sum(sn1a_cube[item]))
            agb.append(np.sum(agb_cube[item]))
    sn2 = np.array(sn2)
    agb = np.array(agb)
    sn1a = np.array(sn1a)

    total_feedback = sn2 + sn1a + agb
    
    
    all_4 = np.vstack((sn2,sn1a,agb,total_feedback))
    if number_of_models_overplotted > 1:
        np.save('output/comparison/elements', elements)
        if os.path.isfile('output/comparison/temp_default.npy'):
            old = np.load('output/comparison/temp_default.npy')
            all_4 = np.dstack((old,all_4))
            np.save('output/comparison/temp_default', all_4)
        else:
            np.save('output/comparison/temp_default', all_4)

    if number_of_models_overplotted > 1:
        medians = np.median(all_4, axis = 2)
        stds = np.std(all_4,axis = 2)
    else:
        medians = all_4
        stds = np.zeros_like(all_4)
    
    if summary_pdf:
        if number_of_models_overplotted == 1:
            plt.clf()
            fig ,ax1 = plt.subplots( figsize = (20,10))
            l1 = ax1.bar(np.arange(len(elements))-0.38,np.divide(sn2,total_feedback), color = 'b' ,label='sn2', width = 0.25)
            l2 = ax1.bar(np.arange(len(elements))-0.13,np.divide(sn1a,total_feedback), color = 'g' ,label='sn1a', width = 0.25)
            l3 = ax1.bar(np.arange(len(elements))+0.12,np.divide(agb,total_feedback), color = 'r' ,label='agb', width = 0.25)
            ax1.set_ylim((0,1))
            plt.yticks(size=30)
            plt.xticks(np.arange(len(elements)), elements, size=30)
            ax1.vlines(np.arange(len(elements)+1)-0.5,0,1)
            ax1.set_ylabel("Fractional Feedback", size=50)
            
            ax2 = ax1.twinx()
            l4 = ax2.bar(np.arange(len(elements)),total_feedback, color = 'k', alpha = 0.2 ,label='total', width = 1)
            ax2.set_yscale('log')
            ax2.set_ylabel('Total Mass Feedback', size=50)
            ax2.tick_params(axis='y', labelsize=30)
            lines = [l1,l2,l3,l4]
            labels = ['SN II', 'SN Ia', 'AGB', 'Total']
            plt.legend(lines, labels,loc='upper right',numpoints=1,fontsize=30).get_frame().set_alpha(0.5)
            plt.savefig('temp/yield_temp_%s.png' %(name_string))
        else:
            plt.clf()
            fig = plt.figure(figsize=(11.69,8.27), dpi=100)
            ax1 = fig.add_subplot(111)
            plt.bar(np.arange(len(elements))-0.05,medians[0], yerr=stds[0],error_kw=dict(elinewidth=2,ecolor='k'), color = 'b' ,label='sn2', width = 0.2)
            plt.bar(np.arange(len(elements))+0.2,medians[1], yerr=stds[1],error_kw=dict(elinewidth=2,ecolor='k'), color = 'g' ,label='sn1a', width = 0.2)
            plt.bar(np.arange(len(elements))+0.45,medians[2], yerr=stds[2],error_kw=dict(elinewidth=2,ecolor='k'), color = 'r' ,label='agb', width = 0.2)
            plt.bar(np.arange(len(elements))+0.45,-medians[2], yerr=stds[2],error_kw=dict(elinewidth=2,ecolor='k'), color = 'r' ,alpha = 0.5, width = 0.2)
            plt.bar(np.arange(len(elements))-0.15,medians[3], yerr=stds[3],error_kw=dict(elinewidth=2,ecolor='y'), color = 'y' ,alpha = 0.1,label = 'total', width = 1)
            plt.ylim((1e-5,1e7))
            plt.xticks(np.arange(len(elements))+0.4, elements)
            plt.vlines(np.arange(len(elements))-0.15,0,1e7)
            plt.yscale('log')
            plt.ylabel("total feedback (arbitrary units)")
            plt.xlabel("element")
            plt.legend(loc='upper right',numpoints=1).get_frame().set_alpha(0.5)
            plt.savefig('temp/yield_temp_%s.png' %(name_string))

    return [probability]

def plotYieldContributions_ALL(elements_to_plot,**kwargs):
    for i,key in enumerate(sorted(kwargs)):
        cube = kwargs[key]
        plot_processes(True,key,cube.sn2_cube,cube.sn1a_cube,cube.agb_cube,
                       elements_to_plot,np.copy(cube),1)




