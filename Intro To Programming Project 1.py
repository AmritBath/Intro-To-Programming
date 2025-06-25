#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1st assignment: forced oscilations 
Created on Sun Oct 15 10:20:14 2023

@author: amritbath

we want to find how many oscillations occur and how long it takes to stop 
oscillating as the system undergoes damping.

We also want to find the fractional decay and the number of oscillations
where the minimum intensity, corresponding to the fraction of decay, is 
larger than the fractional intensity, and the time which these oscillations
correspond to.

"""

# imports numpy and matplotlib.pyplot to allow mathematical calculations and
# graph plotting
import numpy as np
import matplotlib.pyplot as plt


CONSTANT_A0 = 2.0  # defines the constant A0, units: m^-1


def validation(output, sort=None, minimum=None, maximum=None):
    """
    Output a prompt and validate the input's type and value.

    Parameters
    ----------
    output : str
    sort : str, optional
    minimum : float, optional
    maximum : float, optional
    sort : str, optional

    Returns
    -------
    input_ : any

    """
    while True:
        input_ = input(output)
        if sort is not None:
            try:
                input_ = sort(input_)
            except ValueError:
                print("Input type must be", sort.__name__)
                continue
        if minimum is not None and input_ <= minimum:
            print("Input must be greater than", minimum)
        elif maximum is not None and input_ >= maximum:
            print("Input must be less than", maximum)
        else:
            return input_

def get_values():

    """
    Gets user inputs for damping factor, frequency and the minimum intensity,
    including validation checks

    Returns
    -------
    damping_factor_a1: float
        Gets damping factor from user.
    freq : float
        gets frequency from user.
    intensity_min : float
        prompts user to enter the amount of decay as a fraction between 0-1, 
        as a float, demonstrates the percentage decay.

    """

    # gets parameter a1 and checks value range
    damping_factor = validation('\nEnter a value for the damping factor,'
                                   ' a1: ', float, 0.1, 50)

    # gets frequency and checks value range
    freq = validation('\nEnter a value for the frequency: ', float, 1, 200)

    # get minimum fractional intensity, and check it is between 0 and 1
    intensity_min = validation('\nEnter the minimum fractional intensity: ',
                               float, 0, 1)

    return damping_factor, freq, intensity_min

def time():
    """
    gets t for a certian time period so we get a graph which looks continuous

    Returns
    -------
    time_total : numpy.ndarray
        Gives 1000 values between time,start and time,end to give a smooth
        curve, so that we can see the graph decay.

    """
    time_total = np.linspace(0, 2.5, 1000)
    return time_total


def amplitude_calculations(time_total, frequency, damping_factor):
    """

    calculates amplitude


    Parameters
    ----------
    time_total
    frequency
    damping_factor

    Returns
    -------
    amplitude : numpy.ndarray
        calculates amplitude using given equation and assumptions made

    """

    # for light damping, A = cos(wt)*decay constant
    # hence a2 = omega
    omega = 2*np.pi*frequency

    # calculate amplitude using given equation and assumptions made
    amplitude = (1/(CONSTANT_A0 +
                    damping_factor*time_total**2))*np.cos(omega*time_total)
    return amplitude


def intensity_calculations(amplitude):
    """
    calulates intensity and therefore finds fractional intensity

    Parameters
    ----------
    amplitude

    Returns
    -------
    fractional_intensity : numpy.ndarray
        the y axis, represented by the amplitude squared, using the formula
        given and hence finding the fractional intensity using intensity values
        divided by the maximum intensity, giving the fractional values.

    """

    # get intensity
    intensity = amplitude**2

    # get fractional intensity by use of intensity and the first maximum
    fractional_intensity = intensity/intensity[0]

    return fractional_intensity


def finding_n_osc(fractional_intensity, intensity_minimum, time_total):
    """

    finds the total number of oscillations, and the number of oscillations
    where the minimum intensity is above the fractional intensity.

    Parameters
    ----------
    fractional_intensity
    intensity_minimum
    time_total

    Returns
    -------
    number_of_oscillations : int
        total number of oscillations for the entire graph
    times_troughs : numpy.ndarray
        the times at which the minimum points occur
    n_osc_before_t : int
        number of oscillations that occur before t_osc

    """

    number_of_oscillations = 0
    n_osc_before_t = -1
    times_troughs = []
    troughs = []

    # code counts troughs and appends corresponding time and fractional
    # intensity values,

    # also counts the number of oscillations in the entire graph, and
    # number of oscillations where the fractional intensity is larger than
    # the minimum intenstiy

    for i in range(len(time_total)):
        if 0 <= i < 999:
            if (fractional_intensity[i] < fractional_intensity[i+1] and
                    fractional_intensity[i] < fractional_intensity[i-1]):
                number_of_oscillations = number_of_oscillations+1
                troughs.append(fractional_intensity[i])
                times_troughs.append(time_total[i])
            elif fractional_intensity[i] > intensity_minimum:
                if (fractional_intensity[i] > fractional_intensity[i+1] and
                        fractional_intensity[i] > fractional_intensity[i-1]):
                    n_osc_before_t = n_osc_before_t + 1

    return (number_of_oscillations,
            times_troughs, n_osc_before_t)


def plot(time_total, fractional_intensity, intensity_minimum,
         number_of_oscillations, times_troughs, n_osc_before_t):
    """
    plots the fractional intensity on the y, against the time on the x, and
    plots lines for the minimum intensity and t,osc.

    Parameters
    ----------
    time_total
    fractional_intensity
    intensity_minimum
    number_of_oscillations
    times_troughs
    n_osc_before_t

    Returns
    -------
    None.

    """

    # plots graph with time on x axis and the fractional_intensity on y
    plt.plot(time_total, fractional_intensity)

    # labels the axis'
    plt.xlabel('time, t (seconds)')
    plt.ylabel('fractional intensity')

    t_osc = times_troughs[n_osc_before_t]

    # plots the inputed minimum intensity from the user on the graph

    plt.axhline(intensity_minimum, c='g', lw=2, ls=':', label='I, min')

    plt.axvline(t_osc, c='r', lw=2, ls=':', label='t,osc')

    plt.text(t_osc+0.02, fractional_intensity[0], 't, osc', color='red')

    # text is right-aligned
    plt.text(time_total[950], intensity_minimum+0.02, 'I, min',
             horizontalalignment='right', color='green')

    # for any oscillations where the fractional intensity was larger
    # than  fractional intensity then print accordingly, and tells us the
    # n of osc and associated time

    # also tells us the number of complete oscillations in the entire graph and
    # difference with the number of oscillations above the minimum intensity

    if n_osc_before_t > 0:


        print(f'\nThere were {n_osc_before_t} oscillations where the fractional'
              f' intensity was larger than {intensity_minimum}')


        print(f'\nIt took {t_osc:.3f} seconds '
              'to complete these oscillations, to the nearest 3dp')


        print(f'\nIn total, there were  {number_of_oscillations} oscillations, '
              f'this was {number_of_oscillations-n_osc_before_t} more than '
              f'where the fractional intensity was larger than the minimum '
              f'intensity')


    # if no oscillations before the minimum intensity was larger than
    # the fractional intensity then we specify there was 0 oscillations,
    # but returns time of first oscillation.

    else:
        print(f'\nThere were {n_osc_before_t} oscillations where the fractional'
              f' intensity was larger than the minimum intensity, but the first'
              f' oscillation occured at {t_osc}')
        print(f'\nThe total number of oscillations in the whole graph is'
              f'{number_of_oscillations}')


def run():
    damping_factor, frequency, intensity_minimum = get_values()

    time_total = time()

    amplitude = amplitude_calculations(time_total, frequency, damping_factor)

    fractional_intensity = intensity_calculations(amplitude)

    (number_of_oscillations, times_troughs, n_osc_before_t) = finding_n_osc(
        fractional_intensity, intensity_minimum, time_total)


    plot(time_total, fractional_intensity, intensity_minimum,
         number_of_oscillations, times_troughs, n_osc_before_t)

run()
