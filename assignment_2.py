#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:58:10 2023

@author: amritbath
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

# dataset1 = open(' z_boson_data_1.csv', 'r')
# dataset2 = open(' z_boson_data_2.csv', 'r')


# approximate_mz = 90 #GeV/c^2
# approximate_Γz = 3 #GeV
# approximate partial width is = 0.08391 GeV

BOOLEAN = ['y', 'yes', 'n', 'no']
PNK_TXT = '\033[38;5;13m'  # outputs pink text
REG_TXT = '\033[0;0m'  # outputs white text


PARTIAL_WIDTH = 0.08391

TRIAL_VALUES = [90, 3]  # mz  in GeV/c^2 and gamma z in GeV


def validation(output, sort=None, minimum=None, maximum=None, values=None):
    """
    Prompts user for input and validates it based on provided criteria.

    Parameters:
    output (str): Message to display to the user.
    sort (type, optional): Desired type for the input.
    minimum (float, optional): Minimum allowable value.
    maximum (float, optional): Maximum allowable value.
    values (list, optional): Acceptable values for string inputs.

    Returns:
    Any: Validated user input, converted to `sort` type if specified.
    """

    while True:
        input_ = input(output)
        if sort is not None:
            try:
                input_ = sort(input_)
            except ValueError:
                print(PNK_TXT + f'\nType must be {sort.__name__}'
                      + REG_TXT)
                continue
        if minimum is not None and input_ <= minimum:
            print(PNK_TXT + f'\nInput must be > {minimum}'
                  + REG_TXT)
        elif maximum is not None and input_ >= maximum:
            print(PNK_TXT + '\nInput must be < {maximum)}' + REG_TXT)
        elif values is not None and input_.lower() not in values:
            print(PNK_TXT + f'\nInput must be {values[1]} or {values[3]}' +
                  REG_TXT)
        else:
            return input_


def is_float(number):
    """
    Checks if a value can be converted to a float.

    Parameters:
    number (str or number): The value to check.

    Returns:
    bool: True if `number` can be converted to float, False otherwise.
    """

    try:
        float(number)
        return True

    except ValueError:
        return False


def is_positive_float(number):
    """
    Check if a string or number is a positive float.

    Parameters:
    number (str or number): Input to be checked.

    Returns:
    bool: True if `number` is a positive float, False otherwise.
    """

    if is_float(number):
        return float(number) > 0
    return False


def data_from_file(filename):
    """
    Generate a numpy array from a CSV file with specific data validation.

    Parameters:
    filename (file object): File object to read data from.

    Returns:
    ndarray: Numpy array of valid data from the file.
    """
    file_content = filename.readlines()

    # Check if the file is empty or has insufficient data
    if is_data_empty(file_content):
        sys.exit()

    data = []
    for line in file_content:
        entries = line.split(',')
        if (is_positive_float(entries[0]) and is_positive_float(entries[1]) and
                is_positive_float(entries[2])):
            sums = np.array(
                [float(entries[0]), float(entries[1]), float(entries[2])])
            data.append(sums)

    return np.vstack(data)


def is_data_empty(file_content):
    """
    Check if the file content is empty or has insufficient data.

    Parameters:
    file_content (list): List of lines from the file.

    Returns:
    bool: True if the file is empty or has insufficient data, False otherwise.
    """
    if not file_content or len(file_content) <= 5:
        print(PNK_TXT + "File is empty or has insufficient data." + REG_TXT)
        return True
    return False


def read_data(file_name):
    """
    Reads data from a given file and processes it.

    Parameters:
    file_name (str): The name of the file to read.

    Returns:
    list: Processed data from the file.
    """

    with open(file_name, 'r', encoding='utf-8') as dataset:
        data = data_from_file(dataset)
        return data


def data_validate():
    """
    Validates user input to select CSV files, either from the default set of
    Z boson data files given, or specified by the user.

    Returns:
    list: A list of file names selected for data processing.

    Throws:
    SystemExit: If the specified files are not found or are not CSVs.
    """

    files = []
    # if different data files to the default ones are requested, the user
    # can input these and use the code to test them
    if validation('\nUse given data? ', str, values=BOOLEAN) in BOOLEAN[2:]:
        n_files = validation('\nNumber of files to be read: ',
                             int, 0)

        # asks for the number of files
        for iteration in range(n_files):
            user_choice = validation(f'\nEnter file name {iteration+1}: ', str)

            # checks file is the right format
            if not (user_choice.endswith('.csv') or
                    user_choice.endswith('.txt')):
                print(PNK_TXT + f'\n{user_choice} must be txt or csv file'
                      + REG_TXT)
                continue

            # checks if file can be found
            if not os.path.exists(user_choice):
                print(PNK_TXT + f"\n{user_choice} couldn't be found"
                      + REG_TXT)
                sys.exit()

            # adds file names to the array of files
            files.append(user_choice)
        return files

    # If data cannot be opened then returns an error message and exits system
    files = ['z_boson_data_1.csv', 'z_boson_data_2.csv']
    if (not os.path.exists(files[0]) or not
            os.path.exists(files[1])):
        print(PNK_TXT + '\nFiles could not be found.' + REG_TXT)
        sys.exit()
    return files


def data_combine():
    """
    Combines data from multiple CSV files into a single array.

    Returns:
    ndarray: Combined data from the selected files.
    """

    files = data_validate()
    data_total = []
    for _, file_name in enumerate(files):
        data_total.append(read_data(file_name))

    return np.vstack(data_total)


def outlier_filter(input_data, expected_values, std):
    """
    Filters outliers from the data based on a standard deviation threshold.

    Parameters:
    input_data (ndarray): The data to be filtered.
    expected_values (ndarray): Expected values for comparison.
    std (float): Threshold for outlier detection in terms of standard deviation.

    Returns:
    ndarray: Data array after filtering outliers.
    """

    outlier = abs(expected_values -
                  input_data[:, 1]) < (std * input_data[:, 2])
    filtered_data = input_data[np.where(outlier)]
    return filtered_data


def function(best_params, energy):
    """
    Calculate expected cross-section values based on given energy and
    parameters.

    Parameters:
    best_params (list): List of parameters [mass, width].
    energy (float or ndarray): Energy values.

    Returns:
    ndarray: Calculated cross-section values.
    """

    # cross section (=sigma) = (12pi/mass**2) *
    # (energy**2/((energy**2 - mass**2)**2 + mass**2 * width_z**2 ))*
    # partial width **2

    first_part = 12*np.pi/((best_params[0])**2)

    second_part = (energy**2)/((((energy**2)-(best_params[0]**2))**2) +
                               (best_params[0]**2)*(best_params[1]**2))

    last_part = PARTIAL_WIDTH**2

    sigma = first_part*second_part*last_part

    # sigma =(Γee**2)*(12*np.pi/TRIAL_VALUES[0]**2)*(energy**2/
    #   ((energy**2-TRIAL_VALUES[0]**2)**2+TRIAL_VALUES[0]**2*TRIAL_VALUES[1]**2))
    sigma = sigma * 0.3894*10**6
    return sigma


def calc_chi_sq(params, energy, cross_section, uncertainty):
    """
    Calculate chi-squared value for a given set of parameters and data.

    Parameters:
    params (list): Parameters for the function, mass and width.
    energy (ndarray): Array of energy values.
    cross_section (ndarray): Array of cross-section values.
    uncertainty (ndarray): Array of uncertainties.

    Returns:
    float: Chi-squared value.
    """

    observed = cross_section
    expected = function(params, energy)
    residuals = observed - expected
    chi_squared = np.sum((residuals / uncertainty)**2)
    return chi_squared


def plot_data(energy_data, cross_section_data, uncertainty_data, best_params,
              data):
    """
    Plots experimental data, fitted curve, and outliers for Z0 Boson Resonance.

    Parameters:
    energy_data (ndarray): Energy values of the dataset.
    cross_section_data (ndarray): Cross-section values of the dataset.
    uncertainty_data (ndarray): Uncertainty values of the dataset.
    best_params (list): Best-fit parameters for the Z boson.
    data (ndarray): Complete dataset including outliers.
    """

    # set the figure size to be nicely viewed
    plt.figure(figsize=(10, 6))
    # plot the data with the error bars
    plt.errorbar(energy_data, cross_section_data, yerr=uncertainty_data,
                 fmt='o', label='Experimental Data', c='k')

    # plots the outliers and the error bars in a different colour
    x_outliers, y_outliers, err = outliers_to_plt(cross_section_data,
                                                  energy_data,
                                                  uncertainty_data, data)
    plt.errorbar(x_outliers, y_outliers, yerr=err, fmt='o',
                 label='Outliers', c='darkorchid')

    # uses function, which corresponds to the equation given, and find
    # the expected y values for our given energy values
    x_values = np.linspace(max(energy_data), min(energy_data), 1000)
    y_values = function(best_params, x_values)

    # plots the expected curve as the expected fit
    plt.plot(x_values, y_values, label='Fitted Curve', color='hotpink')

    # labels the axis and title
    plt.xlabel('Centre-of-Mass Energy (GeV)')
    plt.ylabel('Cross-Section (mb)')
    plt.title('Fit of $Z_0$ Boson Resonance to Experimental Data')
    plt.legend()
    plt.grid(True)
    plt.savefig('Filtered data fitted to expected data.png', dpi=600)
    plt.show()


def scale(value):
    """
    Determine scaling factor based on a given value.

    Parameters:
    value (float): Input value to determine scaling factor for.

    Returns:
    float: Scaling factor.
    """

    if value <= 20:
        scalar = 1.007
    elif 20 < value <= 70:
        scalar = 1.0005
    elif 70 < value <= 100:
        scalar = 1.0005
    else:
        scalar = 1.000005
    return scalar


def contour_plot(chi2_min, params, energy, cross_section, uncertainty):
    """
    Generate and plot chi-squared contour plot.

    Parameters:
    chi2_min (float): Minimum chi-squared value.
    params (list): Best-fit parameters [mass, width].
    energy (ndarray): Energy data.
    cross_section (ndarray): Cross-section data.
    uncertainty (ndarray): Uncertainty data.

    Returns:
    QuadContourSet: Contour plot object.
    """

    # creates a mesh array of the paramaters and of the chi squared values
    mass_z_mesh, gamma_z_mesh, chi_squared_mesh = mesh_values(params, energy,
                                                              cross_section,
                                                              uncertainty)
    # fits figure size
    fig = plt.figure(figsize=(7, 5))
    plot = fig.add_subplot(111)

    # plots a filled contour plot using the mesh values
    contour = plot.contourf(mass_z_mesh, gamma_z_mesh, chi_squared_mesh,
                            levels=15, zorder=0)

    # labels the axis
    plot.set_xlabel('$m_Z$, boson mass (GeV/$c^2$)', fontsize=12)
    plot.set_ylabel('$Γ_Z$, boson width (GeV)', fontsize=12)

    plot.set_title('Chi-Squared Contour Plot', fontsize=15)

    # creates a contour at chi squared + 1 to find the standard deviation and
    # visualise this on the plot
    std = plot.contour(mass_z_mesh, gamma_z_mesh, chi_squared_mesh, colors='w',
                       linestyles='--', linewidths=1, levels=[chi2_min + 1])

    # labels the value of std
    plot.clabel(std, colors='w')

    # plots the label for the std contour
    dashed_line_text = "Dashed line,\nshows min $\chi^2 + 1$"
    plot.text(0.85, 0.82, dashed_line_text,
              transform=plot.transAxes, fontsize=9,
              verticalalignment='center', horizontalalignment='center',
              color='w', bbox={"facecolor": 'w', "alpha": 0.3,
                               "boxstyle": 'round,pad=0.5'})

    # plots the minimum chi sqared
    plot.scatter(params[0], params[1], c='white', marker='x', s=50,
                 label=f'Min $\u03C7^2$ = {chi2_min:.3}')

    # plots the best fit parameters and lines which draw them to the minimum
    # chi squared from each axis
    plot.axhline(params[1], 0, 0.49, c='w', ls='--')
    plot.axvline(params[0], 0, 0.49, c='w', ls='--')
    plot.text(params[0] * 0.99952, params[1] *
              1.0002, f'{params[1]:.4}', c='w')
    plot.text(params[0] * 1.00001, params[1] *
              0.9932, f'{params[0]:.4}', c='w')

    fig.colorbar(contour)

    plt.legend()

    plt.savefig('contour plot of minimised chi squared.png', dpi=600)

    plt.show()

    return std


def calc_err(stdev):
    """
    Calculate errors in mass and width from contour plot.

    Parameters:
    contour (QuadContourSet): Contour plot object.

    Returns:
    tuple: Errors in mass and width (mass_z_err, gamma_err).
    """

    # use definition of standard deviation from contour plot to find the
    # errors in the best fit paramaters
    ellipse_values = stdev.allsegs[0][0]

    mass_z_err = np.max(
        ellipse_values[:, 0] - np.min(ellipse_values[:, 0])) / 2
    gamma_err = np.max(ellipse_values[:, 1] - np.min(ellipse_values[:, 1])) / 2

    return mass_z_err, gamma_err


def find_best_parameters(parameters, energy, cross_section, uncertainty):
    """
    Find best-fit parameters using chi-squared minimization.

    Parameters:
    parameters (list): Initial guess for parameters.
    energy (ndarray): Energy data.
    cross_section (ndarray): Cross-section data.
    uncertainty (ndarray): Uncertainty data.

    Returns:
    tuple: Best-fit parameters and corresponding chi-squared value.
    """

    # Define fmin
    fit = fmin(calc_chi_sq, parameters,
               args=(energy, cross_section, uncertainty),
               full_output=1, disp=0)

    # Extract the best parameters
    best_params = np.array(fit[0])
    best_chi_squared = float(fit[1])

    return best_params, best_chi_squared


def needed_values(data):
    """
    Extract energy, cross-section, and uncertainty values from data.

    Parameters:
    data (ndarray): Input data array.

    Returns:
    tuple: Arrays of energy, cross-section, and uncertainty.
    """

    energy = data[:, 0]
    cross_section = data[:, 1]  # in mb
    uncertainty = data[:, 2]  # in mb
    return energy, cross_section, uncertainty


def refine(data, params, energy_raw, std):
    """
    Refine parameters by iterative filtering and parameter finding.

    Parameters:
    data (ndarray): Input data array.
    params (list): Initial parameters for refinement.
    energy_raw (ndarray): Raw energy data.
    std (float): Standard deviation threshold for filtering.

    Returns:
    tuple: Refined parameters, chi-squared value, and filtered data.
    """

    # use the outliar filter to remove outliers within a certian number
    # of standard deviations
    filtered_data = outlier_filter(data, function(params, energy_raw), std)

    # find energy cross section and uncertienty
    energy, cross_section, uncertainty = needed_values(filtered_data)

    # find the best parameters and best chi squared using these values
    best_params, best_chi_squared = find_best_parameters(params, energy,
                                                         cross_section,
                                                         uncertainty)

    return best_params, best_chi_squared, energy, cross_section, uncertainty


def mesh_values(params, energy, cross_section, uncertainty):
    """
    Generates mesh grid arrays for mz, gamma_z, and chi-squared calculations.

    Parameters:
    params (list): Parameters for calculations (mass, width).
    energy (ndarray): Energy data.
    cross_section (ndarray): Cross-section data.
    uncertainty (ndarray): Uncertainty data.

    Returns:
    tuple: Mesh grids for mz, gamma_z, and chi-squared.
    """

    mass_z_range = np.linspace(params[0]/scale(params[0]),
                               params[0]*scale(params[0]), 100)
    gamma_z_range = np.linspace(params[1]/scale(params[1]),
                                params[1]*scale(params[1]), 100)

    # create mesh grid using the linspace values
    mass_z_mesh, gamma_z_mesh = np.meshgrid(mass_z_range, gamma_z_range)

    # define the chi squared mesh to be empty but of the same size
    chi_squared_mesh = np.zeros(mass_z_mesh.shape)

    # calculate the chi squared for each index of the mesh array based on the
    # paramaters in the corresponding index for the paramater mesh arrays
    for i in range(mass_z_mesh.shape[0]):
        for j in range(mass_z_mesh.shape[1]):
            mesh_params = [mass_z_mesh[i, j], gamma_z_mesh[i, j]]
            chi_squared_mesh[i, j] = calc_chi_sq(mesh_params,
                                                 energy,
                                                 cross_section,
                                                 uncertainty)

    return mass_z_mesh, gamma_z_mesh, chi_squared_mesh


def red_chi_sq(min_chi_squared, filtered_data, params):
    """
    Calculates the reduced chi-squared value.

    Parameters:
    min_chi_squared (float): Minimum chi-squared value from the fit.
    filtered_data (ndarray): Data after filtering outliers.
    params (list): Parameters used in the fit.

    Returns:
    float: Reduced chi-squared value.
    """

    # χ2/Ndof
    # Ndof= number of data points, N, - number of free parameters in the fit

    reduced_chi_squared = min_chi_squared/(len(filtered_data)-len(params))

    return reduced_chi_squared


def lifetime_calc(width, uncertainty_width):
    """
    Calculates the lifetime of the Z boson and its uncertainty.

    Parameters:
    width (float): Width of the Z boson.
    uncertainty_width (float): Uncertainty in the width.

    Returns:
    tuple: Lifetime of the Z boson and its uncertainty.
    """

    # ΓZ= ħ/ τZ
    # in natural units, ħ=1 so ΓZ= 1 / τZ
    # 1 GeV−1 = 6.582×10−25 s

    conversion_to_seconds = 6.582*10**-25

    lifetime = (1/width)*conversion_to_seconds
    lifetime_uncertienty = (uncertainty_width/width) * lifetime

    return lifetime, lifetime_uncertienty


def outliers_to_plt(cross_sec, energy, err, data):
    """
    Identify and separate outliers from a dataset.

    Parameters:
    cs_useful (ndarray): Cross-section data of the reference dataset.
    energy_useful (ndarray): Energy data of the reference dataset.
    err_use (ndarray): Uncertainty data of the reference dataset.
    cs (ndarray): Cross-section data of the comparison dataset.
    energy (ndarray): Energy data of the comparison dataset.
    err (ndarray): Uncertainty data of the comparison dataset.

    Returns:
    tuple: Arrays of energy, cross-section, and uncertainty for outliers.
    """

    energy_raw = data[:, 0]  # in GeV already

    # find the best parameters using the guess of 90 and 3

    (_, _, energy_useful, cross_sec_useful,
     err_use) = refine(data, TRIAL_VALUES, energy_raw, 1000)

    # Create sets of tuples for comparison
    data_useful = set(zip(energy_useful, cross_sec_useful, err_use))
    data_all = set(zip(energy, cross_sec, err))

    # Find outlier tuples
    outliers = data_useful.difference(data_all)

    # Separate the outliers into individual arrays
    outliers_x = np.array([x for x, _, _ in outliers])
    outliers_y = np.array([y for _, y, _ in outliers])
    outliers_err = np.array([e for _, _, e in outliers])

    return outliers_x, outliers_y, outliers_err


def main():
    """
    Main function to process data and determine Z boson properties.
    """

    data = np.array(data_combine())
    energy_raw = data[:, 0]  # in GeV already

    # find the best parameters using the guess of 90 and 3

    (best_params, best_chi_squared, _, _,
     _) = refine(data, TRIAL_VALUES, energy_raw, 1000)

    # use the best paramaters found to refine the guess and find the
    # better paramaters to use, and use these to remove the outliers

    (best_params, best_chi_squared, energy, cross_section,
     uncertainty) = refine(data, best_params, energy_raw, 3)
    (best_params, best_chi_squared, energy, cross_section,
     uncertainty) = refine(data, best_params, energy_raw, 3)

    # set the best paramater variables
    mass_z_best, gamma_z_best = best_params

    # plotting(energy,cross_section,uncertainty,fit)

    plot_data(energy, cross_section, uncertainty, best_params, data)

    # use the contour_plot function in order to display contours and return
    # the standard deviation
    std = contour_plot(best_chi_squared, best_params, energy,
                       cross_section, uncertainty)

    # find the uncertienty using the standard deviation of the contour plot
    mass_z_uncertienty, width_z_uncertienty = calc_err(std)
    reduced_chi_sq = red_chi_sq(best_chi_squared, energy, best_params)

    # use uncertienty on best fit paramaters to find lifetime
    lifetime, lifetime_uncertienty = lifetime_calc(gamma_z_best,
                                                   width_z_uncertienty)

    # print findings
    print(f'\nMin chi-squared: {best_chi_squared:.3f}',
          f'\nCorresponding reduced chi-squared: {reduced_chi_sq:.3f}')

    print("\nBest-fit parameters:\nmz, mass of Z boson:",
          f"{mass_z_best:.4} ± {mass_z_uncertienty:.2f}GeV/c^2\nΓz,",
          f"the width of the Z boson: {gamma_z_best:.4} ±",
          f"{width_z_uncertienty:.2}GeV")

    print(f'\nLifetime of Z boson: {lifetime:.2e} ±',
          f'{lifetime_uncertienty:.1e} seconds')


main()
