'''
Header      : reactors_v1.py
Created     : 22.06.2022
Modified    : 25.06.2022
Author      : Vladimir Kozlov, kozlov.vlr@yandex.ru
Description : Merging CSTReactor model with plug-Flow Reactor model.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Species:
    '''
    Describes chemical species

    Methods
    ----------
    .Cp(T: float)
        Calculates Heat Capacity (constant Pressure) at specified temperature
    '''
    def __init__(self, name: str, MW: float, coeffs_Cp: list):
        '''
        :param name: Species Name
        :param MW: [g/mol] Molar Weight
        :param coeffs_Cp: Coefficients for Heat Capacity correlation
        '''
        self.name = name
        self.MW = MW
        self.Cp_coeffs = coeffs_Cp

    def Cp(self, T: float):
        '''
        Returns Specific Heat Capacity [J/(mol*K)] of pure component at specified Temperature

        :param T: [K] Temperature
        :return: [J/(mol*K)] Specific Heat Capacity
        '''
        return 4.1887 * (self.Cp_coeffs[0] + self.Cp_coeffs[1] * T
                         + self.Cp_coeffs[2] * T ** 2 + self.Cp_coeffs[3] * T ** 3)


class Reaction:
    '''
    Describes kinetic reactions

    Methods
    ----------
    .rate(T: float)
        Calculates Reaction Rate at specified temperature with Arrhenius Law
    '''
    def __init__(self, name: str, reagents: list[Species], stoic: list[float], dH: float, k0: float, E0: float):
        '''
        :param name: Reaction Name
        :param reagents: List of all reagents (same order as in equation)
        :param stoic: List of stoichiometric coefficients for listed components (same order as reagents). Negative values
        for reagents (left side of equation), positive for products (right side)
        :param dH: [kJ/mol] Heat of Reaction (Implemented temporarily before calculation as difference of enthalpies of formation
        would be introduced)
        :param k0: Reaction Rate Constant (Arrhenius Parameter)
        :param E0: [kJ/mol] Activation Energy
        '''
        self.name = name
        self.reagents = reagents
        compnameslist = list()
        for comp in reagents:
            compnameslist.append(comp.name)
        self.stoic = dict(zip(compnameslist, stoic))
        self.dH = dH
        self.k0 = k0
        self.E0 = E0

    def rate(self, T: float, conc: dict):
        '''
        Returns Reaction Rate at specified temperature

        :param T: [K] Temperature
        :param conc: [kmol/m3] Concentrations of components
        :return: Reaction Rate
        '''
        '''Multiplier concidering contribution of concentrations to Arrhenius Law'''
        mult = 1
        for comp in self.reagents:
            if self.stoic[comp.name] < 0:
                # Now calculates with concentrations in [kmol/m3]!
                mult = mult * ((conc[comp.name]) ** abs(self.stoic[comp.name]))
        # Needs to be revised!
        '''
        equation used: v = k * prod([I] ^ i)
            where k = k0 * exp(E0 / R / T)
        '''
        return self.k0 * np.exp(-self.E0 / 8.3144 / T) * abs(mult)


def get_mixcp(comps: list[Species], mol_frac: dict, T: float):
    '''
    Returns Specific Heat Capacity [J/(mol*K)] of mixture at specified Temperature

    :param comps: List of components in mixture
    :param mol_frac: [mol. frac.]List of components molar fractions for listed components
    :param T: [K] Temperature
    :return: [J/(mol*K)] Specific Heat Capacity
    '''
    sum = 0
    for comp in comps:
        sum += mol_frac[comp.name] * comp.Cp(T)
    return sum


def get_molfrac(conc : dict):
    '''
    Calculates Molar Fractions [mol. frac.] of components in mixture from given concentrations [kmol/m3]
    :param conc: [kmol/m3] List of components concentrations
    :return: [mol. frac.] Molar Fractions
    '''
    molfrac = dict()
    for i in conc.keys():
        molfrac[i] = conc[i] / np.array(list(conc.values())).sum()
    return molfrac


class ReactorModel:
    '''
    Describes Chemical Reactors in terms of Continuous Stirred Tank Reactor and Plug-Flow Reactor models

    Methods
    ----------
    .Simulation(self, init_T: float, press: float, init_C: dict, res_t: float, int_t: float, log: bool)
        Performs integration along time axis and returns concentrations and molar fractions of reagents in mixture
        as well as mixture temperature at different moments of time
    '''
    def __init__(self, compset: list [Species], rxnset: list[Reaction], rctrtype: str):
        '''
        :param compset: Set of components including reagents for all reactions in reactor
        :param rxnset: Set of reactions occurring in reactor
        :param rctrtype: Type of reactor ('cstrctr' or 'pfrctr')
        '''
        self.compset = compset
        self.rxnset = rxnset
        self.rctrtype = rctrtype

    def Simulation(self, init_T: float, press: float, init_C: dict, res_t: float, int_t: float, log: bool):
        '''
        Performs  integration along time axis with Euler method for reactions listed

        :param init_T: [K] Initial temperature
        :param press: [MPa] Reaction Pressure
        :param init_C: [kmol/m3] Initial concentrations of components (including products)
        :param res_t: [s] Residence time (for CSTReactor)
        :param int_t: [s] Integration time
        :param log: Select if tabular results for each timestep is required
        :return: Concentrations [kmol/m3], molar fraction [mol. frac.] and temperatre [K]
        '''
        # init_T - Initial Temperature [K]
        # press - Reaction Pressure [MPa]
        # init_C - Initial Concentrations of reagents [dict of float]
        # res_t - Residence Time [s]
        # int_t - Integration Time [s]
        # log - Option whether or not return all history of calculations [bool]
        t = np.array([0])  # Actual Time [s]
        dt = 0.1  # Integration Timestep [s]
        T = init_T  # Actual Temperature [K]
        act_C = init_C.copy()  # Actual Components Concentrations [kmol/m3]
        x = get_molfrac(init_C)  # Actual Molar Fractions [mol. frac.]
        calc_hist = pd.DataFrame()  # Storage for calculations results

        # Method to perform Reactor integration by Euler method
        while t[-1] <= int_t:
            # Writing down Concentrations from previous step
            # Temporary DataFrame is created to list all components in one row
            temp_df = pd.DataFrame()
            for comp in self.compset:
                temp_df['C - {}'.format(comp.name)] = [act_C[comp.name]]
                temp_df['x - {}'.format(comp.name)] = [x[comp.name]]
            temp_df['T'] = T
            temp_df['t'] = [t[-1]]
            # Merging temporary DataFrame with main one
            calc_hist = calc_hist.append(temp_df)
            # Calculating reaction mixture Heat Capacity
            Cp_mix = get_mixcp(compset, x, T)  # [J/(mol*K)]
            if self.rctrtype == 'cstrctr':
                # For Continuous Stirred Tank Reactors
                for rxn in self.rxnset:
                    # Performing Heat Balance for each reaction at a time
                    T = T + dt * ((init_T - T) / res_t + (-rxn.dH * 1000 * rxn.rate(T, act_C))
                                  * 0.00845 * T / press / Cp_mix)  # [K]
                    for comp in rxn.reagents:
                        # Performing Material Balance for each component at a time
                        act_C[comp.name] = act_C[comp.name] + dt * ((init_C[comp.name] - act_C[comp.name]) / res_t
                                                                    + rxn.stoic[comp.name] * rxn.rate(T, act_C))  # [kmol/m3]
            elif self.rctrtype == 'pfrctr':
                # For Plug-Flow Reactors
                for rxn in self.rxnset:
                # Performing Heat Balance for each reaction at a time
                    T = T + dt * (-rxn.dH * 1000 * rxn.rate(T, act_C)) * 0.00845 * T / press / Cp_mix  # [K]
                    for comp in rxn.reagents:
                        # Performing Material Balance for each component at a time
                        act_C[comp.name] = act_C[comp.name] + dt * (rxn.stoic[comp.name] * rxn.rate(T, act_C))  # [kmol/m3]
            else:
                print('WARNING! Check the reactor type key input (must be "cstrct" or "pfrctr"')
            # Calculating molar fractions of components
            x = get_molfrac(act_C)  # [mol.frac.]
            t = np.append(t, t[-1] + dt)
        calc_hist = calc_hist.set_index('t', drop=True)
        if log:
            # If log writing option is selected whole DataFrame of calclulations history is returned
            return calc_hist
        else:
            # Otherwise only parameters at last iteration is returned
            return act_C, x, T


def plot_results(results: pd.DataFrame):
    '''
    Plots results of reactor simulation. Input DataFrame must be in format defined in ReactorModel.Simulation

    :param results: Tabular results of ReactorModel.Simulation
    '''
    fig, axes = plt.subplots(2, 1)
    axes[0].set_title('Composition and Temperature of Reaction Mixture vs. Time')
    axes[0].grid()
    axes[0].set_ylabel('Molar Fraction, [mol. frac.]')
    for param in results.columns:
        if 'x - ' in param:
            axes[0].plot(results.index, results[param], label= param)
    axes[0].legend()
    axes[1].set_ylabel('Temperature, [K]')
    axes[1].set_xlabel('Time, [s]')
    axes[1].plot(results.index, results['T'])
    plt.grid()
    plt.show()
    return 1


'''---------------------------------------------------------------------------'''
'''Initializing Data'''
print('Initializing calculations...')
'''Setting up constants'''
Rprime = 0.00845  # Universal Gas Constant [(m3*MPa)/(kmol*K)]
'''Setting up reagents for both reactions'''
compA = Species('n-C8H18', 114.23, [-1.456, 0.1842, -0.0001002, 0.00000002115])
compB = Species('i-C8H18', 114.23, [-2.201, 0.1877, -0.0001051, 0.00000002316])
compC = Species('C4H10', 58.12, [2.266, 0.07913, -0.00002647, -0.000000000674])
compD = Species('C4H8', 56.11, [-0.715, 0.08436, -0.00004754, 0.00000001066])
compset = [compA, compB, compC, compD]
'''Setting up reactions themselves'''
rxn1 = Reaction('n-C8H18 Isomerisation', [compA, compB], [-1, 1], -7.03, .12, 94.2)
rxn2 = Reaction('i-C8H18 Cracking', [compB, compC, compD], [-1, 1, 1], 85.89, .80, 81.2)
rxnset = [rxn1, rxn2]
'''Setting up initial conditions'''
tau = 6  # Residence Time [s]
t_end = 10  # Integration Time [s]
P = .1013  # Reaction Pressure [MPa]
T0 = 620  # Initial Temperature [K]
comp_C0 = dict({'n-C8H18' : .0388, 'i-C8H18' : 0, 'C4H10' : 0, 'C4H8' : 0}) # Component Initial Concentrations [kmol/m3]

print('Starting calculations...')
'''Creating CSTReactor model'''
cstreactor = ReactorModel(compset, rxnset, 'pfrctr')
'''Integrating through CTReactor model'''
calc_hist = cstreactor.Simulation(T0, P, comp_C0, tau, t_end, True)
print('\tCalculations completed successfully!')

'''Saving results to .xlsx file'''
filepath = os.path.join(os.getcwd(), 'cstr_results.xlsx')
print('Saving Results to {}...'.format(filepath))
calc_hist.to_excel(filepath)


'''Plotting diagrams in matplotlib'''
print('Plotting graphs...')
plot_results(calc_hist)

'''Done)'''
print('done')