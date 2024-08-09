import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
import pandas as pd
import os

'''
Comments:
The code is written in kW and kWh
A mixed integer linear programming approach is adopted to identify the optimal size for PV area and battery capacity to minimize total costs.
Total costs are intended in this code as the sum of annualized investment costs (associated to PV and battery) and yearly operational costs deriving from electricity imports from the grid.
The optimization is carried out with a 1-hour resolution and 1-year time horizon.
The techno-economic data used in the code are adopted from the database:
    https://sweet-cross.ch/data/energy-tech-parameters/2024-02-27/
'''

plt.close('all')

## Parameters

# Techno-economic parameters
eff_b_ch = 0.98  # Battery charging efficiency
eff_b_disch = 0.97  # Battery discharging efficiency
eff_b_sd = 1-0.125/24  # Battery self-discharge efficiency, average value from table
UP_b = 470  # Unit price of the battery in CHF/kWh, average price from table 
lifetime_b = 10  # Lifetime of the battery in years, average lifetime from table
# SOC range imposed from Depth of Discharge between 80 to 100%
SOC_max = 1.0  # Max state of charge of the battery to increase lifetime
SOC_min = 0.0  # Min state of charge of the battery to increase lifetime
eff_PV = 0.205  # Efficiency of the PV system, average value from table
UP_PV = 0.94  # Unit price for the PV installation in CHF/kW_peak; average price from table
lifetime_PV = 25  # lifetime PV, average value from table

# Design Constraints
P_imp_max = 10  # Maximum imported power
Area_PV_max = 75  # Maximum available area for PV in m2
C_b_max = 50  # Maximum battery capacity to be installed in kWh 
C_b_min = 0  # Minimum battery capacity in kWh

# Boundary conditions and Design Constraints
c_el = 0.30  # Cost to import electricity in CHF/kWh
current_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_path, 'residential_hour_profile_EMPA.csv')
df = pd.read_csv(file_path)
P_dem = df['Electricity demand [J](Hourly) '].values / 3.6e6  # Convert from J to kWh
G_irradiance = df['G [W](msquared)'].values  # Solar irradiance in W/m2

# Artificial parameters
epsilon = 1e-4  # Small value to avoid numerical issues
max_sim_time = 5 * 60  # Max simulation time in seconds
hours_in_year = 8760

## Optimization model with cvxpy

# Decision variables
P_imp = cp.Variable(hours_in_year)  # Power imported from the grid
P_ch = cp.Variable(hours_in_year)  # Power charged to the battery
P_disch = cp.Variable(hours_in_year)  # Power discharged from the battery
E_b = cp.Variable(hours_in_year + 1)  # Energy in the battery at each time step
C_b = cp.Variable()  # Battery Capacity
Area_PV = cp.Variable()  # PV area in m2

# PV power generation
P_PV = (G_irradiance * Area_PV * eff_PV) / 1000

# Objective function
cost_inv = C_b * UP_b / lifetime_b + (Area_PV * 1000 * eff_PV) * UP_PV / lifetime_PV  # Investment cost for the battery and PV
cost_op = cp.sum(P_imp * c_el)  # Operational costs
cost = cost_inv + cost_op

# Constraints
constraints = [
    E_b[1:] == E_b[:-1] * eff_b_sd + P_ch * eff_b_ch - P_disch / eff_b_disch,  # Energy conservation for the battery storage
    P_disch <= E_b[:-1],  # Constraint for discharged power lower than available power
    E_b[:-1] <= SOC_max * C_b,  # Energy content in the battery below maximum capacity
    E_b[:-1] >= SOC_min * C_b,  # Energy content above minimum state of charge
    P_disch / eff_b_disch + P_imp + P_PV >= P_dem + P_ch,  # Energy balance for the energy system
    E_b[hours_in_year] == E_b[0],  # Periodicity
    E_b >= 0,  # Non-negative energy in battery
    P_imp >= 0,  # Non-negative power import
    P_imp <= P_imp_max,  # imported power below maximum
    P_ch >= 0,  # Non-negative charging power
    P_disch >= 0,  # Non-negative discharging power
    C_b >= 0,  # Non-negative battery capacity
    C_b <= C_b_max,  # Maximum battery capacity
    Area_PV >= 0,  # Non-negative PV area
    Area_PV <= Area_PV_max  # maximum PV area
]

# Problem definition
problem = cp.Problem(cp.Minimize(cost), constraints)

# Solve the problem
problem.solve(solver=cp.ECOS, max_iters=max_sim_time)

if problem.status == cp.OPTIMAL:
    print("Optimal solution found.")
    print(f"Total cost: {cost.value:.2f} CHF")
    print(f"Optimized battery capacity: {C_b.value:.2f} kWh")
    print(f"Optimized PV area: {Area_PV.value:.2f} m²")
    
    # Calculate Levelized Cost of Energy (LCOE)
    LCOE = cost.value / np.sum(P_dem)
    print(f"Levelized Cost of Energy (LCOE): {LCOE:.4f} CHF/kWh")
    
    # Calculate Self-Sufficiency
    En_dem_year = np.sum(P_dem)
    En_grid_year = np.sum(P_imp.value)
    self_sufficiency = (En_dem_year - En_grid_year) / En_dem_year
    print(f"Self-Sufficiency: {self_sufficiency:.4%}")
    
else:
    print(f"Solver ended with status: {problem.status}")

#######################################################
## Plotting ##

# figure for electricity demand
plt.figure(figsize=(10, 6))
plt.plot(P_dem, label='Electricity Demand (kWh)', color='blue', linewidth=0.1)
plt.xlabel('Time (hours)', fontsize=18)
plt.ylabel('Electricity Demand (kWh)', fontsize=18)
plt.title('Electricity Demand Over One Year', fontsize=20)
plt.legend(fontsize=16)
plt.grid(True)
plt.ylim(0, 3)  # Set y-axis limits
# Increase the font size for the numbers on both axes
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

# figure for solar irradiance
plt.figure(figsize=(10, 6))
plt.plot(G_irradiance, label='Solar Irradiance (W/m²)', color='orange', linewidth=0.5)
plt.xlabel('Time (hours)', fontsize=18)
plt.ylabel('Solar Irradiance (W/m²)', fontsize=18)
plt.title('Solar Irradiance Over One Year', fontsize=20)
plt.legend(fontsize=16)
plt.xlim(0, 8760)  # Set x-axis limits
plt.ylim(0, 1000)  # Set y-axis limits
plt.grid(True)

# Increase the font size for the numbers on both axes
plt.tick_params(axis='both', which='major', labelsize=14)

plt.show()


################## results

# Post-optimization: Extracting values for plotting
energy_demand = P_dem
power_supplied = P_disch.value + P_imp.value + P_PV.value - P_ch.value
import_power = P_imp.value
charging_power = P_ch.value
discharging_power = P_disch.value
battery_energy = E_b.value
pv_power = P_PV.value

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(energy_demand, label='Energy Demand')
plt.plot(power_supplied, label='Power Supplied')
plt.plot(import_power, label='Imported Power')
plt.plot(charging_power, label='Charging Power')
plt.plot(discharging_power, label='Discharging Power')
plt.plot(battery_energy, label='Battery Energy')
plt.plot(pv_power, label='PV Power')
plt.xlabel('Time (hours)')
plt.ylabel('Energy (kWh) / Power (kW)')
plt.legend()
plt.title('Energy and Power over Time')
plt.grid(True)
plt.show()
