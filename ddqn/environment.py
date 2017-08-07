"""
EnergyPlus fmu simulation as reinforcement learning environment.
"""


from pyfmi import load_fmu
import os
import sys
import numpy as np
import pandas as pd
from time import sleep
import datetime
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import pdb
import gc



logger = logging.getLogger(__name__)
"""
EnergyPlus FMU model for co-simulation.
"""

class EP(gym.Env):

    reward_range = (-np.inf, np.inf)
    action_space = None
    observation_space = None

    def __init__(self,
                 energy_plus_file,
                 state_size,
                 action_space,
                 solar_panel_area=50,
                 solar_panel_percent_active=0.9,
                 solar_panel_efficiency=0.9,
                 inverter_efficiency=0.9,
                 battery_discharge_efficiency=0.9,
                 battery_charge_efficiency=0.9,
                 battery_capacity=5000,
                 simulation_length=10):

        self.model = load_fmu(energy_plus_file,
                              log_file_name='log_file.txt',
                              kind='auto')


        self.sim_duration = 86400
        self.numSteps = 144
        self.opts = self.model.simulate_options()
        self.opts['ncp'] = self.numSteps

        self.battery = Battery(capacity=battery_capacity,
                          charge_efficiency=battery_charge_efficiency,
                          discharge_efficiency=battery_discharge_efficiency)
        self.solar = SolarPanel(area=solar_panel_area,
                           f_active=solar_panel_percent_active,
                           eta_cell=solar_panel_efficiency,
                           eta_inv=inverter_efficiency)
        self._seed()
        self.action_space = action_space
        self.state = None
        self.state_size = state_size
        self.observation_space = None
        self.date =  datetime.date(2017, 1, 1)
        self.counter = 0
        self.store = pd.DataFrame({})
        self.time = self.model.time


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
#         assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state

        start_time = self.counter * self.sim_duration
        final_time = (self.counter + 1) * self.sim_duration

        t = np.linspace(start_time,
                        final_time,
                        self.numSteps + 1,
                        dtype=np.int32)



        battery_action = self.action_space[action] # replace with action
        battery_flag = self.battery.step(battery_action)

#         avail_manager_action = action[:, 1]  # skipping the HVAC control for now
        avail_manager_action = np.random.randint(1, 2, 145)

        ut = np.transpose(np.vstack((t, np.repeat(avail_manager_action, 1))))
        input_object = ('Q', ut)

        res = self.model.simulate(start_time=start_time,
                             final_time=final_time,
                             input=input_object,
                             options=self.opts)
        self.time = self.model.time


        df = pd.DataFrame({})
        for key in self.model.get_model_variables().keys():
            df[key] = res[key]

        df['managerAction'] = avail_manager_action
        df['battery_flag'] = battery_flag.insert(0, 0)
        df[self.battery.id] = self.battery.get_history()['state'][self.counter * self.numSteps:
                                                        (self.counter + 1) * self.numSteps + 1]
        df['charge_discharge'] = battery_action
        df['solar_generation'] = self.solar.step(res['directSolarRad'])
        df['substation_electiricty'] = res['totalDemand'] + df['charge_discharge'] - df['solar_generation']
        df.index = pd.DatetimeIndex(pd.date_range(str(self.date.month) + '/' + str(self.date.day) + '/' +
                                                  str(self.date.year), periods=145, freq='10min'))



        #df.to_hdf('./save/store.h5', 'table', append=True)

        self.store = self.store.append(df)
        self.date = self.date + datetime.timedelta(days=1)

        self.state = df[['outdoorDbTemp', 'solar_generation',
                         self.battery.id, 'totalDemand']]
        self.state['weekday'] = self.date.weekday()
        self.state['month'] = self.date.month

        #updating the counter
        self.counter += 1

        reward = - np.clip(df['substation_electiricty'], -1, 1)
        done = np.zeros(reward.shape[0]) > 1
        done[-1] = True
        return self.state.values, reward, done

    def _reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05,
                                      size=(145, self.state_size))
        return np.array(self.state)

    def _close(self):
        # self.model.terminate()
        gc.collect()

    def stored_data():
        return pd.read_hdf('./save/store.h5', 'table')



class Space(object):
    """Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.
    """

    def sample(self):
        """
        Uniformly randomly sample a random element of this space
        """
        self.state = np.random.uniform(low=-0.05, high=0.05,
                                      size=(145, 4))
        return np.array(self.state)

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        return True

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n



class Battery(object):
    """
    class for simulating battery.
    """

    idCounter = 0

    def __init__(self, id='battery', capacity=100, discharge_efficiency=0.7,
                 charge_efficiency=0.7,
                 timeStep=6,
                 minimum_charge=0):
        # self.id = id + '_' + str(Battery.idCounter)
        # Battery.idCounter += 1
        self.id = id
        self.discharge_efficiency = discharge_efficiency
        self.charge_efficiency = charge_efficiency
        self.capacity = capacity
        self.state = [capacity]
        self.timeStep = 60/timeStep
        self.time = []
        self.minimum_charge = minimum_charge
        self.action = [0]
        self.current = [0]

    def charge(self, action):

            self.state.append(self.state[-1] +
                              self.charge_efficiency * action)
            self.current.append(self.charge_efficiency * action)
            self.action.append(action)

    def discharge(self, action):

            self.state.append(self.state[-1] +
                              action / self.discharge_efficiency)
            self.current.append(action / self.discharge_efficiency)
            self.action.append(action)


    def step(self, actions):
        flag = []
        for action in actions:
            if len(self.time) == 0:
                self.time.append(0)
                continue

            self.time.append(self.time[-1] + 60/self.timeStep)
            if (action > 0 and
               (self.state[-1] + action * self.charge_efficiency)
               < self.capacity):

                self.charge(action)
                flag.append(1)

            elif (action < 0 and
                  (self.state[-1] + action / self.discharge_efficiency)
                  > self.minimum_charge):

                self.discharge(action)
                flag.append(1)

            else:
                self.state.append(self.state[-1])
                self.action.append(0)
                self.current.append(0)
                flag.append(0)

        return flag

    def get_state(self):
        return {'time': self.time[-1], 'state': self.state[-1]}

    def get_history(self):
        return {'time': np.array(self.time), 'state': np.array(self.state),
                'action': np.array(self.action),
                'current': np.array(self.current)}


class SolarPanel(object):

    """
    solar panel module.
    """

    def __init__(self, area, f_active, eta_cell, eta_inv, timeStep=6):

        """
        Arguments:
            area -- integer, net area of surface.
            f_active -- fraction of surface area with solar cell.
            eta_cell -- module conversion efficiency.
            eta_inv -- DC to AC conversion efficiency.
        """
        self.area = area
        self.f_active = f_active
        self.eta_cell = eta_cell
        self.eta_inv = eta_inv
        self.time = []
        self.timeStep = 60/timeStep

    def step(self, g_t):
        """
        Arguments:
            g_t -- numpy ndarray, Total solar radiation incident on PV array
                   [W/m2]

        Return:
            P --  numpy ndarray, Electrical power produced by photovoltaics [W]
        """
        P = self.area * self.f_active * self.eta_cell * self.eta_inv * g_t

        return P
