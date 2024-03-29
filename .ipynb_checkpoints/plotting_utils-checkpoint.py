import numpy as np
import pandas as pd

import statsmodels.api
import scipy.stats

import bokeh.io
import bokeh.plotting

def plot_regression_comparison(df, df_averages, x, spir, morphs, samples, slope_comp, x_ppc, circle_size=7, circle_alpha=0.3, line_width=3, line_alpha=1, size=300):

    y_min, y_max = np.min(df[morphs].values), np.max(df[morphs].values)
    x_min, x_max = np.min(df['log mass (g)'].values), np.max(df['log mass (g)'].values)
    intercept1 = first_intercept(slope_comp, x_max, y_min)
    line_scale = (y_max - y_min)/10
    
    p = bokeh.plotting.figure(width=size, height=size,
                          y_range=(y_min-0.2, y_max+0.2),
                          x_range=(x_min-0.2, x_max+0.2)#, y_axis_label=morphs, x_axis_label='log mass',
                         )
    
    [p.line(generate_line(intercept=i, slope=slope_comp, point=x_max)[0],
            generate_line(intercept=i, slope=slope_comp, point=x_max)[1], color='grey', alpha=0.3)
     for i in line_scale*np.array(range(30))+intercept1]

    #p.legend.location = 'bottom_right'
    p.xgrid.visible = False
    p.ygrid.visible = False

    
    ############################################
    ############################################
    ############################################

    p_alpha, color = (0.2, 'grey')
    x = x_ppc
    # y-values of each point
    y = samples.posterior_predictive['y_ppc'][0].values

    for interval in [[2.5, 97.5], [10, 90]]:
        low, high = np.percentile(y, interval, axis=0)
        p1 = np.append(x, x[::-1])
        p2 = np.append(low, high[::-1])
        p.patch(p1, p2, alpha=p_alpha, color=color)
    
    
    p_alpha, color = (0.5, 'steelblue')
    # x-values
    x = np.linspace(x.min(), x.max(), 200)
    # y-values of each point
    y = np.outer(samples.posterior['a'].values[0], x) + np.stack([samples.posterior['b'].values[0]]*200, axis=1)
    
    for interval in [[2.5, 97.5],[45, 55]]:# [10, 90], [20, 80]]:
        low, high = np.percentile(y, interval, axis=0)
        p1 = np.append(x, x[::-1])
        p2 = np.append(low, high[::-1])
        if interval[0] == 45:
            p.patch(p1, p2, alpha=1, color=color)
        else:
            p.patch(p1, p2, alpha=p_alpha, color=color)
    
    ############################################
    ############################################
    ############################################

    x = df_averages.sort_values('mass (g)').loc[df_averages['spiracle'] == spir, 'log mass (g)'].values
    y = df_averages.sort_values('mass (g)').loc[df_averages['spiracle'] == spir, morphs].values


    Y = df.sort_values('mass (g)').loc[df['spiracle'] == spir, morphs].values
    X = df.sort_values('mass (g)').loc[df['spiracle'] == spir, 'log mass (g)'].values
    X = statsmodels.api.add_constant(X)
    model = statsmodels.api.OLS(Y,X)
    results = model.fit()
    results.params


    #p.circle(df.sort_values('mass (g)').loc[df['spiracle'] == spir, 'log mass (g)'].values,
    #         df.sort_values('mass (g)').loc[df['spiracle'] == spir, morphs].values, size=circle_size, alpha=circle_alpha, fill_color='white', line_color='white')

    #p.line(np.linspace(x.min(), x.max(), 200),
    #       results.params[1]*np.linspace(x.min(), x.max(), 200) + results.params[0],
    #       line_width=line_width, color='white', alpha=line_alpha)#, line_dash='dashed')


    Y = y
    X = x
    X = statsmodels.api.add_constant(X)
    model = statsmodels.api.OLS(Y,X)
    results = model.fit()
    results.params

    #p.line(np.linspace(x.min(), x.max(), 200),
    #       np.median(samples.posterior['a'].values.ravel())*np.linspace(x.min(), x.max(), 200) + np.median(samples.posterior['b'].values.ravel()),
    #       line_width=8, color='steelblue', line_alpha=line_alpha)

    #p.line(np.linspace(x.min(), x.max(), 200),
    #       results.params[1]*np.linspace(x.min(), x.max(), 200) + results.params[0],
    #       line_width=line_width, color='white', line_alpha=line_alpha)#, line_dash='dashed')

    p.circle(x, y, size=circle_size, alpha=circle_alpha, color='steelblue')
    
    p.title.text = spir + ' slope 95% CI: ' + str([round(j, 3) for j in np.percentile(samples.posterior['a'].values.ravel(), [2.5, 97.5])])

    p.output_backend='svg'
    return(p)

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""
    # Set up array of indices to sample from
    inds = np.arange(len(x))

    # Initialize samples
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)
    bs_σ_reps = np.empty(size)

    # Take samples
    for i in range(size):
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        p, residuals, rank, singular_values, rcond = np.polyfit(bs_x, bs_y, deg=1, full=True)
        bs_slope_reps[i], bs_intercept_reps[i] = p#np.polyfit(bs_x, bs_y, deg=1, full=True)
        bs_σ_reps[i] = np.sqrt(residuals/(len(x)-2))

    return bs_slope_reps, bs_intercept_reps, bs_σ_reps

def make_CIs(df, to_plot, spiracle, q=[2.5, 97.5]):
    bs_slope_reps, bs_intercept_reps, bs_σ_reps  = draw_bs_pairs_linreg(
                        df.loc[(df['spiracle'] == spiracle), 'log mass (g)'].values,
                        df.loc[(df['spiracle'] == spiracle), to_plot].values,
                                                        size=10000)
    
    p, residuals, rank, singular_values, rcond = np.polyfit(df.loc[(df['spiracle'] == spiracle), 'log mass (g)'].values, 
                                  df.loc[(df['spiracle'] == spiracle), to_plot].values, deg=1, full=True)
    slope, intercept = p
    σ = np.sqrt(residuals/(len(df.loc[(df['spiracle'] == spiracle), 'log mass (g)'].values)-2))
        
    return(np.percentile(bs_slope_reps, q), np.percentile(bs_intercept_reps, q), np.percentile(bs_σ_reps, q), slope, intercept, σ)

def generate_line(slope, intercept, point=0, move=100):
    x1 = point-move
    x2 = point+move
    y1 = slope*x1 + intercept
    y2 = slope*x2 + intercept
    return (x1, x2), (y1, y2)

def first_intercept(slope, x_max, y_min):
    return(y_min-slope*x_max)

def make_plot(df, to_plot, slope_comp, n_cols=4):
    
    def draw_bs_pairs_linreg(x, y, size=1):
        """Perform pairs bootstrap for linear regression."""
        # Set up array of indices to sample from
        inds = np.arange(len(x))

        # Initialize samples
        bs_slope_reps = np.empty(size)
        bs_intercept_reps = np.empty(size)

        # Take samples
        for i in range(size):
            bs_inds = np.random.choice(inds, len(inds))
            bs_x, bs_y = x[bs_inds], y[bs_inds]
            bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, deg=1)

        return bs_slope_reps, bs_intercept_reps

    plots = []

    for spiracle in ['S', 'T', '1', '2', '3', '4', '5', '6']:

        y_min, y_max = np.min(df[to_plot].values), np.max(df[to_plot].values)
        x_min, x_max = np.min(df['log mass (g)'].values), np.max(df['log mass (g)'].values)
        intercept1 = first_intercept(slope_comp, x_max, y_min)
        line_scale = (y_max - y_min)/10

        p = bokeh.plotting.figure(width=230, height=230,
                                  y_range=(y_min-0.2, y_max+0.2),
                                  x_range=(x_min-0.2, x_max+0.2)
                                 )
        [p.line(generate_line(intercept=i, slope=slope_comp, point=x_max)[0],
                generate_line(intercept=i, slope=slope_comp, point=x_max)[1], color='grey', alpha=0.3)
         for i in line_scale*np.array(range(30))+intercept1]

        p.scatter('log mass (g)', to_plot,
                  source = df.loc[(df['spiracle'] == spiracle)])

        #p.legend.location = 'bottom_right'
        p.xgrid.visible = False
        p.ygrid.visible = False
        
        slope, intercept = np.polyfit(df.loc[(df['spiracle'] == spiracle), 'log mass (g)'].values, 
                              df.loc[(df['spiracle'] == spiracle), to_plot].values, deg=1)
        x = np.array([x_min, x_max])
        y = slope * x + intercept

        p.line(x, y)
        
        bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(
                        df.loc[(df['spiracle'] == spiracle), 'log mass (g)'].values,
                        df.loc[(df['spiracle'] == spiracle), to_plot].values,
                                                        size=10000)
        
        p.title.text = spiracle + ' slope 95% CI: ' + str([round(j, 3) for j in np.percentile(bs_slope_reps, [2.5, 97.5])])

        # x-values
        x = np.linspace(x_min, x_max, 200)

        # y-values of each point
        y = np.outer(bs_slope_reps, x) + np.stack([bs_intercept_reps]*200, axis=1)

        # Compute the 2.5th and 97.5th percentiles
        low, high = np.percentile(y, [2.5, 97.5], axis=0)
        
        p1 = np.append(x, x[::-1])
        p2 = np.append(low, high[::-1])

        p.patch(p1, p2, alpha=0.5)
        p.output_backend='svg'
        plots.append(p)
        

    bokeh.io.show(bokeh.layouts.gridplot(plots,ncols=n_cols))