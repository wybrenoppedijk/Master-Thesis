# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 14:12:07 2022
https://stackoverflow.com/questions/51926684/plotting-sum-of-two-sinusoids-in-python
@author: dgeco
"""

import numpy as np
import time
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick



def plotting_example_sinusoidal(): 
    ### Simple example showing how to plot three signals, where the third is a combination of the other two, in the same graph ###
    
    left, width = 0.1, 0.8
    rect1 = [left, 0.65, width, 0.25]  # left, bottom, width, height
    rect2 = [left, 0.4, width, 0.25]
    rect3 = [left, 0.1, width, 0.3]
    
    fig = plt.figure(figsize=(10, 6))
    
    ax1 = fig.add_axes(rect1) 
    ax2 = fig.add_axes(rect2, sharex=ax1)
    ax3 = fig.add_axes(rect3, sharex=ax1)
    
    x = np.linspace(0, 6.5*np.pi, 200)
    y1 = np.sin(x)
    y2 = np.sin(2*x)
    
    ax1.plot(x, y1, color='b', lw=2)
    ax2.plot(x, y2, color='g', lw=2)
    ax3.plot(x, y1+y2, color='r', lw=2)
    
    ax3.get_xaxis().set_ticks([])
    
    for ax in [ax1, ax2, ax3]:
        ax.hlines(0, 0, 6.5*np.pi, color='black')
        for key in ['right', 'top', 'bottom']:
            ax.spines[key].set_visible(False)
    
    plt.xlim(0, 6.6*np.pi)
    ax3.text(2, 0.9, 'Sum signal', fontsize=14)



def plotting_initial_signals(input_type): 
    ### Combination of three motor signals, and denoting two different types of flexibility activation ###
  
    T=5
    D=5
    N=5
    shift = 1/4   # number of cycles to shift (1/4 cycle in your example). From zero to one represents a whole cycle.
    x = np.linspace(0, T*N, 10000, endpoint=False)
    y1 = 2.5 + 2.5*signal.square(2 * np.pi * (1/T) * x + 2*shift*np.pi, duty=0.5)
    y2 = 2.5 + 2.5*signal.square(2 * np.pi * (1/T) * x + 3*shift*np.pi, duty=0.5)
    y3 = 2.5 + 2.5*signal.square(2 * np.pi * (1/T) * x + shift*np.pi, duty=0.5)
    # y1 = 2.5 + 2.5*signal.square(2 * np.pi * (1/T) * x + 2*shift*np.pi, duty=0.25)
    # y2 = 2.5 + 2.5*signal.square(2 * np.pi * (1/T) * x + 3*shift*np.pi, duty=0.15)
    # y3 = 2.5 + 2.5*signal.square(2 * np.pi * (1/T) * x + shift*np.pi, duty=0.85)
    
    
    shift2 = 1/8
    y1_s = 2.5 + 2.5*signal.square(2 * np.pi * (1/T) * x + 2*(shift2)*np.pi)
    y2_s = 2.5 + 2.5*signal.square(2 * np.pi * (1/T) * x + 6*(-shift2)*np.pi)
    y3_s = 2.5 + 2.5*signal.square(2 * np.pi * (1/T) * x + shift2*np.pi)
    
    y12 = np.concatenate((y1[:5000], y2[5000:]))
    print(len(y12))
    
    # Setting up figure
    
    left, width = 0.1, 0.8
    rect1 = [left, 0.675, width, 0.25]  # left, bottom, width, height
    # rect2 = [left, 0.4, width, 0.20]
    rect3 = [left, 0.1, width, 0.5]
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_axes(rect1) 
    # ax2 = fig.add_axes(rect2, sharex=ax1)
    ax3 = fig.add_axes(rect3, sharex=ax1)
    
    # Setting up plots
    
    # Normal operation
    def normal_plot():
        ax1.plot(x, y12)#, 'blue', x, y2, 'red', x, y3, 'black')#, color='m')
        ax3.plot(x, y12 + y2 + y3, 'k')
    
    
    # Shift in power alone
    def power_shift_plot():
        # ax1.plot(x, y1, 'blue', x, y2, 'red', x, y3, 'm', x, 0.80*y1, 'b--', x, 0.80*y2, 'r--', x, 0.80*y3, 'm--')#, color='b')
        ax1.plot(x, y1, 'b--', x, 0.80*y1, 'b')
        y_power = y1+y2+y3 - 0.80*(y1+y2+y3)
        ax3.plot(x, y1+y2+y3, 'k--', x, 0.80*(y1+y2+y3), 'k', x, y1+y2+y3 - 0.80*(y1+y2+y3), 'c')
        ax3.fill_between(x, y_power, color='cyan')
    
    
    # # Shift in activation time
    def time_shift_plot():
        ax1.plot(x, y1, 'b--', x, y1_s, 'b')#, color='m')
        ax3.plot(x, y1 + y2 + y3, 'k--', x, (y1_s + y2_s + y3_s), 'k')
        ax3.fill_between(x,5, color='cyan')
    
    if input_type == 'normal':
        normal_plot()
    elif input_type == 'power':
        power_shift_plot()
    else:
        time_shift_plot()
    
    
    #ax3.plot(x, y1_s + y2_s + y3_s, 'm--')
    
    #plt.plot(x, y1+y2+y3, x, y1+y2+y3 - 0.80*(y1+y2+y3), 'c--', x, (y1_s + y2_s + y3_s), 'm--')
    #plt.plot(x, )
    # ax1.set_xticklabels([])
    # plt.ylim(0, 20.0)
    plt.xlim(0, T*N)
    # fig.subplots_adjust(wspace=0.2, hspace=0.2)



def plotting_normal_dist_10_30_100(): 
    ### Combination of ten, thirty and a hundred motor signals with normal distributions ###
    ### Highlighting flexibility of full-stop over small period of time ###
  
    
    T=5
    D=5
    N=5
    x = np.linspace(0, T*N, 10000, endpoint=False)
    
    left, width = 0.1, 0.8
    rect1 = [left, 0.70, width, 0.25]  # left, bottom, width, height
    rect2 = [left, 0.4, width, 0.25]
    rect3 = [left, 0.1, width, 0.25]
    
    fig = plt.figure(figsize=(6, 6))
    
    ax1 = fig.add_axes(rect1) 
    ax2 = fig.add_axes(rect2, sharex=ax1)
    ax3 = fig.add_axes(rect3, sharex=ax1)
    
    
    ### 10 motors### 10 motors ##### 10 motors ##### 10 motors ##### 10 motors ##
    motor_distribution_1 = np.random.normal(5, 0.75, 10)
    time_distribution_1 = np.random.normal(0, 10, 10)
    
    time_distribution_small_1 = np.random.normal(1, 0.5, 10)
    
    
    y_normal_1 = 0
    for idx, val in enumerate(motor_distribution_1):    
        y_normal_1 = y_normal_1 + (motor_distribution_1[idx] + motor_distribution_1[idx]*signal.square(2 * np.pi * (time_distribution_small_1[idx]/T) * x + 2*time_distribution_1[idx]*np.pi))
    
    # ax1.plot(x, y1, 'b', x, y1_s, 'b--')#, color='m')
    ax1.plot(x, y_normal_1, 'k')
    ax1.fill_between(x,min(y_normal_1), color='cyan')
    plt.ylim(0,(max(y_normal_1)))
    # ax1.ylabel('Aggregated load [kW]')
    
    
    ### 30 motors ##### 30 motors ##### 30 motors ##### 30 motors ##### 30 motors ##
    motor_distribution_2 = np.random.normal(5, 0.75, 30)
    time_distribution_2 = np.random.normal(0, 10, 30)
    
    time_distribution_small_2 = np.random.normal(1, 0.5, 30)
    
    y_normal_2 = 0
    for idx, val in enumerate(motor_distribution_2):    
        y_normal_2 = y_normal_2 + (motor_distribution_2[idx] + motor_distribution_2[idx]*signal.square(2 * np.pi * (time_distribution_small_2[idx]/T) * x + 2*time_distribution_2[idx]*np.pi))
    
    # ax1.plot(x, y1, 'b', x, y1_s, 'b--')#, color='m')
    ax2.plot(x, y_normal_2, 'k')
    ax2.fill_between(x,min(y_normal_2), color='cyan')
    plt.ylim(0,(max(y_normal_2)))
    # ax2.ylabel('Aggregated load [kW]')
    
    # print (y_normal_2)
    
    ### 100 Motors ##### 100 Motors ##### 100 Motors ##### 100 Motors ##### 100 Motors ##
    motor_distribution = np.random.normal(5, 0.75, 100)
    time_distribution = np.random.normal(0, 10, 100)
    
    time_distribution_small = np.random.normal(1, 0.5, 100)
    
    y_normal = 0
    for idx, val in enumerate(motor_distribution):    
        y_normal = y_normal + (motor_distribution[idx] + motor_distribution[idx]*signal.square(2 * np.pi * (time_distribution_small[idx]/T) * x + 2*time_distribution[idx]*np.pi))
    
    # print(type(y_normal), y_normal)
    # print(min(y_normal))
    
    # ax1.plot(x, y1, 'b', x, y1_s, 'b--')#, color='m')
    ax3.plot(x, y_normal, 'k')
    ax3.fill_between(x,min(y_normal), color='cyan')
    plt.ylim(0,(max(y_normal)))
    
    plt.xlabel('Time period, td 0 10 td_small 1 0.5')
    plt.ylabel('                                                                                     Aggregated load [kW]')
    
    ax1.set_title('10 motors')
    x_axis = ax1.axes.get_xaxis()
    x_axis.set_visible(False)
    ax2.set_title('30 motors')
    x_axis = ax2.axes.get_xaxis()
    x_axis.set_visible(False)
    ax3.set_title('100 motors')
    
    print(sum(motor_distribution))
    
    #ax3.plot(x, y1_s + y2_s + y3_s, 'm--')
    
    #plt.plot(x, y1+y2+y3, x, y1+y2+y3 - 0.80*(y1+y2+y3), 'c--', x, (y1_s + y2_s + y3_s), 'm--')
    #plt.plot(x, )
    # ax1.set_xticklabels([])
    # plt.ylim(0, 20.0)
    plt.xlim(0, T*N)
    # fig.subplots_adjust(wspace=0.2, hspace=0.2)





def calc_occurrences_1_to_100(): 
    ### Combination of one to a hundred motor signals with normal distributions ###
    ### Computing total flexibility and proportion to available power ###
   
    T=5
    D=5
    N=5
    x = np.linspace(0, T*N, 10000, endpoint=False)
    
    left, width = 0.15, 0.8
    rect1 = [left, 0.135, width, 0.75]  # left, bottom, width, height
    # rect2 = [left, 0.4, width, 0.25]
    # rect3 = [left, 0.1, width, 0.25]
    
    fig = plt.figure(figsize=(7, 4))
    
    ax1 = fig.add_axes(rect1) 
    # ax2 = fig.add_axes(rect2, sharex=ax1)
    # ax3 = fig.add_axes(rect3, sharex=ax1)
    
    duty_cycle     = 0.50
    total_simul    = 5
    total_motors   = 50
    motors         = list(range(1, total_motors+1))
    min_power      = []
    min_power_dict = {}
    min_power_pct  = []
    min_power_90th = []
    min_power_95th = []


    for m in motors:        
        
        total_one_motor = time.time()
        
        for s in range(0, total_simul):
            
            ### Creating normal distribution specs for motor power, period, and time of activation for each motor ###
            motor_distribution      = np.random.normal(2.5, 0.75, m)    # 2.5 because we are adding 2.5 + 2.5 to shift the square wave up.
            time_distribution       = np.random.normal(0, 10, m)
            time_distribution_small = np.random.normal(1, 0.5, m)
            
            ### Creating square signals for each motor, given specs above ###
            motor_profile = 0
            for idx, val in enumerate(motor_distribution):    
                motor_profile = motor_profile + (motor_distribution[idx] + motor_distribution[idx]*signal.square(2 * np.pi * (time_distribution_small[idx]/T) * x + 2*time_distribution[idx]*np.pi, duty=duty_cycle))
            
            # print(type(motor_profile), motor_profile)
            # print(min(motor_profile))
            # if s == 0:
            #     # print(s, m, min(motor_profile))
            #     print(s, m)
                
            ### Identifying the minimum occurrence across distribution of motor profiles (absolute minimum) ###
            if int(s) == 0:
                # print(s, m, min(motor_profile))
                min_power.append(min(motor_profile))
                min_power_pct.append(min(motor_profile)/(2*sum(motor_distribution)))
                min_power_dict[m] = [min(motor_profile)/(2*sum(motor_distribution))]
            elif min(motor_profile) < min_power[m-1]:
                min_power[m-1]     = min(motor_profile)
                min_power_pct[m-1] = min(motor_profile)/(2*sum(motor_distribution))
                min_power_dict[m].append(min(motor_profile)/(2*sum(motor_distribution)))
        
        # print(min_power_dict[m], type(min_power_dict[m]))
        
        if m == 1:
            min_power_05th = [0]
            min_power_15th = [0]
            min_power_25th = [0]
        else:        
            min_power_05th.append(np.percentile(min_power_dict[m],  5))
            min_power_15th.append(np.percentile(min_power_dict[m], 15))
            min_power_25th.append(np.percentile(min_power_dict[m], 25))
            
        print("I took", time.time() - total_one_motor, "seconds to finish computing", total_simul, "simulations, for a total of", m,"aggregated motors.")
    
    # pars, cov = curve_fit
    
    ax1.plot(motors, min_power_pct, 'k', label='Min. avail. power')
    ax1.plot(motors, min_power_05th, 'c', label='95th percentile')
    ax1.plot(motors, min_power_15th, 'r', label='85th percentile')
    ax1.plot(motors, min_power_25th, 'b', label='75th percentile')
    # ax2.plot(motors, min_power_90th)
    # ax3.plot(motors, min_power_95th)
    
    

    ### Curve fitting ### Curve fitting ### Curve fitting ### Curve fitting ###
    def f(x,a,b,c):
        return a * np.exp(-b*(1/x)) + c #https://www.wolframalpha.com/input?i=exp+%28-1%2Fx%29
        
    params, extras = curve_fit(f, motors, min_power_05th)
    a,b,c = params
    
    ax1.plot(motors, f(np.asarray(motors),a,b,c), ':', label='Curve fitting to 95th percentile')
    
    
    
    plt.legend()
    
    ax1.set_title('Available active power flexibility with %1.2f' %duty_cycle + ' duty cycle \n (%i independent simulations for $\it{m}$ motors)' %total_simul)
    # ax1.set_title('Available active power flexibility with 0.25 duty cycle \n (%i independent simulations for $\it{m}$ motors)' %total_simul)
    plt.xlabel('Number of motors $\it{m}$')
    plt.ylabel('Minimum available power at any time \n [pct. of total power]')

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xlim(0, total_motors)
    
    


if __name__ == "__main__":
    plotting_example_sinusoidal()
    plotting_initial_signals('normal')
    plotting_initial_signals('power')
    plotting_initial_signals('time')
    plotting_normal_dist_10_30_100()
    calc_occurrences_1_to_100()




