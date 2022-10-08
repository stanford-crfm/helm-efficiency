def setup_plotting_params():
    import matplotlib
    import seaborn as sns
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    sns.set_style('ticks')
    import matplotlib.font_manager
    for font in matplotlib.font_manager.findSystemFonts(
        '/Users/deepak/Library/Fonts/'):
        matplotlib.font_manager.fontManager.addfont(font)
    font = {
        'font.family':'Roboto',
        'font.weight': 1000,
        'font.size': 12,
    }
    sns.set_style(font)
    paper_rc = {
        'lines.linewidth': 3,
        'lines.markersize': 10,
    }
    sns.set_context("paper", font_scale=3,  rc=paper_rc)
    current_palette = sns.color_palette()