"""
plot_style.py
-------------
Central per-plot style configuration.  Edit the PLOT_STYLES table below —
one entry per figure number.  Each plotting function loads its entry by key
before creating the figure; you never need to touch the plotting scripts.

Figure index
------------
 1  profile_resolution.pdf        paper_figures.plot_resolution_profile
 2  Zoom1.png                      (manually generated)
 3  phase_combined.pdf             paper_figures.plot_phase_diagrams
 4  energy_evolution.pdf           disk_analysis/plot_energy_evolution
 5  profile_shell_mass_accretion   paper_figures.plot_shell_mass_accretion
 6  profile_density.pdf            paper_figures.plot_profile_overlay
 7  profile_kinematics_disk.pdf    paper_figures.plot_kinematic_radial_profiles
 8  profile_kinematics_wide.pdf    paper_figures.plot_kinematic_radial_profiles
 9  combined_density.pdf           paper_figures.plot_grid_combined
10  toomre_Q_merged.pdf            paper_figures.plot_toomre_Q_merged
11  profile_toomre_Q.png           paper_figures.plot_profile_overlay
12  sink_count_history_loglog.pdf  disk_analysis/plot_sink_history
13  combined_Btor.pdf              paper_figures.plot_grid_combined
14  combined_Bz.pdf                paper_figures.plot_grid_combined
15  phase_Bfield.pdf               paper_figures.plot_bfield_phase
16  profile_mass_to_flux.pdf       paper_figures.plot_profile_overlay
17  mass_evolution.pdf             disk_analysis/plot_mass_evolution  (fig_a)
18  mass_evolution_individual_loglog  disk_analysis/plot_sink_history
19  mass_evolution_rates.pdf       disk_analysis/plot_mass_evolution  (fig_b)
20  ./light/IMF2.png               (manually generated)

Parameters (all per-figure)
----------------------------
font_size            : fallback for any text element not listed below
tick_label_size      : numbers on tick marks (x and y)
axis_label_size      : x and y axis label text
legend_font_size     : legend text

tick_major_size      : major tick length (pts)
tick_minor_size      : minor tick length (pts)
tick_major_width     : major tick stroke width
tick_minor_width     : minor tick stroke width
axes_linewidth       : box / spine / frame stroke width
line_width           : default plot-line stroke width

marker_size          : scatter / line marker diameter (pts)
legend_marker_scale  : legend marker size relative to plot markers
                       (raise this when dots look too small in the legend)
legend_handle_length : length of line handles shown in legend (em units)
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class PlotStyle:
    # fonts
    font_size:            float = 30
    tick_label_size:      float = 27
    axis_label_size:      float = 33
    legend_font_size:     float = 20
    # tick geometry
    tick_major_size:      float = 8
    tick_minor_size:      float = 4
    # line widths
    tick_major_width:     float = 2.4
    tick_minor_width:     float = 1.6
    axes_linewidth:       float = 2.0
    line_width:           float = 2.0
    # markers
    marker_size:          float = 6.0
    legend_marker_scale:  float = 2.0
    legend_handle_length: float = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# Edit here — one entry per figure
# ══════════════════════════════════════════════════════════════════════════════
PLOT_STYLES = {

    'fig_1': PlotStyle(   # profile_resolution.pdf
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 8,
        tick_minor_size      = 6,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 1.0,
        legend_handle_length = 2.0,
    ),

    #'fig_2': PlotStyle(   # Zoom1.png  (manually generated — entry here for completeness)
    #    font_size            = 30,
    #    tick_label_size      = 27,
    #    axis_label_size      = 33,
    #    legend_font_size     = 20,
    #    tick_major_size      = 8,
    #    tick_minor_size      = 4,
    #    tick_major_width     = 2.4,
    #    tick_minor_width     = 1.6,
    #    axes_linewidth       = 2.0,
    #    line_width           = 2.0,
    #    marker_size          = 6.0,
    #    legend_marker_scale  = 2.0,
    #    legend_handle_length = 2.0,
    #),

    'fig_3': PlotStyle(   # phase_combined.pdf
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_4': PlotStyle(   # energy_evolution.pdf
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_5': PlotStyle(   # profile_shell_mass_accretion.pdf
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_6': PlotStyle(   # profile_density.pdf
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 12,
        tick_minor_size      = 6,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_7': PlotStyle(   # profile_kinematics_wide.pdf
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_8': PlotStyle(   # combined_density.pdf
        font_size            = 20,
        tick_label_size      = 20,
        axis_label_size      = 20,
        legend_font_size     = 20,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_9': PlotStyle(  # toomre_Q_merged.pdf
        font_size            = 20,
        tick_label_size      = 20,
        axis_label_size      = 20,
        legend_font_size     = 20,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 1.0,
        legend_handle_length = 2.0,
    ),

    'fig_10': PlotStyle(  # profile_toomre_Q.png
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_11': PlotStyle(  # sink_count_history_loglog.pdf
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_12': PlotStyle(  # combined_Btor.pdf
        font_size            = 20,
        tick_label_size      = 20,
        axis_label_size      = 20,
        legend_font_size     = 20,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_13': PlotStyle(  # combined_Bz.pdf
        font_size            = 20,
        tick_label_size      = 20,
        axis_label_size      = 20,
        legend_font_size     = 20,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_14': PlotStyle(  # phase_Bfield.pdf
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 1.5,
        legend_handle_length = 2.0,
    ),

    'fig_15': PlotStyle(  # profile_mass_to_flux.pdf
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 12,
        tick_minor_size      = 6,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 3.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_16': PlotStyle(  # mass_evolution.pdf
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 12,
        tick_minor_size      = 6,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_17': PlotStyle(  # mass_evolution_individual_loglog.pdf
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    'fig_18': PlotStyle(  # mass_evolution_rates.pdf
        font_size            = 25,
        tick_label_size      = 25,
        axis_label_size      = 25,
        legend_font_size     = 25,
        tick_major_size      = 8,
        tick_minor_size      = 4,
        tick_major_width     = 2.4,
        tick_minor_width     = 1.6,
        axes_linewidth       = 2.0,
        line_width           = 2.0,
        marker_size          = 6.0,
        legend_marker_scale  = 2.0,
        legend_handle_length = 2.0,
    ),

    #'fig_19': PlotStyle(  # ./light/IMF2.png  (manually generated — entry for completeness)
    #    font_size            = 30,
    #    tick_label_size      = 27,
    #    axis_label_size      = 33,
    #    legend_font_size     = 20,
    #    tick_major_size      = 8,
    #    tick_minor_size      = 4,
    #    tick_major_width     = 2.4,
    #    tick_minor_width     = 1.6,
    #    axes_linewidth       = 2.0,
    #    line_width           = 2.0,
    #    marker_size          = 6.0,
    #    legend_marker_scale  = 2.0,
    #    legend_handle_length = 2.0,
    #),

}


# ══════════════════════════════════════════════════════════════════════════════
# Internal machinery — no need to edit below this line
# ══════════════════════════════════════════════════════════════════════════════

def apply_style(name_or_style) -> PlotStyle:
    """Apply a named style ('fig_N' key) or a PlotStyle object to rcParams."""
    style = PLOT_STYLES[name_or_style] if isinstance(name_or_style, str) else name_or_style
    plt.rcParams.update({
        'font.size':             style.font_size,
        'axes.labelsize':        style.axis_label_size,
        'axes.titlesize':        style.axis_label_size,
        'xtick.labelsize':       style.tick_label_size,
        'ytick.labelsize':       style.tick_label_size,
        'legend.fontsize':       style.legend_font_size,
        'xtick.major.size':      style.tick_major_size,
        'xtick.minor.size':      style.tick_minor_size,
        'ytick.major.size':      style.tick_major_size,
        'ytick.minor.size':      style.tick_minor_size,
        'xtick.major.width':     style.tick_major_width,
        'xtick.minor.width':     style.tick_minor_width,
        'ytick.major.width':     style.tick_major_width,
        'ytick.minor.width':     style.tick_minor_width,
        'axes.linewidth':        style.axes_linewidth,
        'lines.linewidth':       style.line_width,
        'lines.markersize':      style.marker_size,
        'legend.markerscale':    style.legend_marker_scale,
        'legend.handlelength':   style.legend_handle_length,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
    })
    return style
