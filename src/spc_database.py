"""
Functions for Parsing the SPC Convective Mode Database

Shawn Murdzek
sfm5282@psu.edu
Date Created: 25 March 2021
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import requests
import xarray as xr
import geopy.distance as gd


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def add_datetime(df):
    """
    Updates the 'date' column in the input DataFrame by combining the 'date' and 'time' columns

    Parameters
    ----------
    df : pandas DataFrame
        Input DataFrame of events from the SPC Convective Mode Database

    Returns
    -------
    df : pandas DataFrame
        DataFrame with an updated 'date' column that combines the 'date' and 'time' columns

    """

    datetimes = []
    for i in range(len(df)):
        time = df['time'].iloc[i]
        datetimes.append(df['date'].iloc[i] + pd.Timedelta('%d hour' % time.hour) + 
                         pd.Timedelta('%d minute' % time.minute))
    df = df.assign(date=datetimes)
    
    return df


def read_spc_database(fname, filter_param=True, remove_duplicates=True, update_date=True,
                      rm_supercell=True):
    """
    Read the SPC Convective Mode Database and save as a Pandas DataFrame.

    Parameters
    ----------
    fname : string
        SPC Convective Mode Database file name (.xlsx)
    filter_param : boolean, optional
        Option to remove rows with missing environmental parameters
    remove_duplicates : boolean, optional
        Option to only retain the most severe storm report within 185 km and +/- 3 hours
    update_date : boolean, optional
        Option to update the 'date' column to be datetime objects
    rm_supercell : boolean, optional
        Option to only retain right-moving supercells not associated with tropical cyclones

    Returns
    -------
    case_df : pandas DataFrame
        Post-processed DataFrame with all cases
    sigtor_df : pandas DataFrame
        Post-processed DatFrame with only nontornadic and significantly tornadic cases

    """
    
    raw_df = pd.read_excel(fname)

    # Remove cases with errors
    
    case_df = raw_df[np.isnan(raw_df['error?'])]
    
    # Only retain right-moving supercells and remove cases associated with tropical cyclones
    
    if rm_supercell:
        case_df = case_df[case_df['super RM'] == 1]
        case_df = case_df[np.isnan(case_df['TC'])]
    
    # Only keep cases with environmental parameters (some missing values are also set to -10000)
    
    param = ['ml_cape', 'ml_cin', 'ml_lcl', 'ml_lfc', 'mu_cape', 'mu_cin', 'mu_lcl', 'mu_lfc', 
             'sb_cape', 'sb_cin', 'cape0_3km', 'dn_cape', 'shr8', 'shr6', 'shr3', 'shr1', 'eff_shr',
             'srh3', 'srh1', 'eff_srh', 'eff_base', 'scp_eff', 'stp', 'STP T03 >0', 'lr700_500', 
             'lr850_500', 'lr0_3km', 'tmp_sfc', 'dwp_sfc', 'rh_sfc', 'pw']
    
    if filter_param:
        for p in param:
            case_df = case_df[np.logical_not(np.isnan(case_df[p]))]
            case_df = case_df[case_df[p] > -9999]
            
    # Update date column
    
    if update_date:
        case_df = add_datetime(case_df)
    
    # Remove duplicate cases (i.e., those within 185 km and 3 hours of each other)
    
    if remove_duplicates:
        
        case_df.reset_index(drop=True, inplace=True)
        
        # Create a case severity column
        
        severity = np.array(case_df['magnitude'].values)
        for t, term in zip(['TORNADO', 'HAIL', 'WIND'], [3000, 2000, 1000]):
            idx = np.where(case_df['type'] == t)[0]
            severity[idx] = severity[idx] + term
        
        # Loop through each case with 185sighail + 185sigwind + 185tor > 0
        # Check for cases within 3 hours
        # Using the cases within 3 hours, check for cases within 185 km
        # Out of the resulting pool of cases, only keep the strongest case, with TOR > HAIL > WIND
        
        idx_drop = []
        cand = np.where(case_df['185sighail'] + case_df['185sigwind'] + case_df['185tor'] > 0)[0]
        visited = np.zeros(len(case_df))
        incr = pd.Timedelta(3, unit='h')
        
        for j in cand:
            if not visited[j]:
                time = case_df['date'].iloc[j]
                time_neighbors = np.where(np.logical_and(case_df['date'] <= time + incr,
                                                         case_df['date'] >= time - incr))[0]
                maxmag = 0
                idxmax = np.nan
                for idx in time_neighbors:
                    d = gd.geodesic((case_df['slat'].iloc[j], case_df['slon'].iloc[j]), 
                                    (case_df['slat'].iloc[idx], case_df['slon'].iloc[idx])).km
                    if d <= 185.0 and idx not in idx_drop:
                        if severity[idx] > maxmag:
                            maxmag = severity[idx]
                            if idxmax == idxmax:
                                idx_drop.append(idxmax)
                                visited[idxmax] = 1
                            idxmax = idx
                        else:
                            idx_drop.append(idx)
                            visited[idx] = 1
                        
        case_df.drop(index=idx_drop, inplace=True)
        case_df.reset_index(drop=True, inplace=True)
        
    # Create dataset that only contains significantly tornadic and null events
    
    sigtor_df = case_df[np.logical_not(np.logical_and(case_df['type'] == 'TORNADO',
                                                      case_df['magnitude'] <= 1))]
    
    # Reset indices
    
    case_df.reset_index(inplace=True)
    sigtor_df.reset_index(inplace=True)
    
    return case_df, sigtor_df


def read_cust_database(fname, drop_yrs=[2013]):
    """
    Read data from SPC database with custom sounding parameters computed using 
    create_snd_database.py

    Parameters
    ----------
    fname : string
        SPC Convective Mode Database file name with custom parameters (.xlsx)
    drop_yrs : list of strings
        Years to drop from the database (defaul is 2013 b/c there is only tornado data for that year)

    Returns
    -------
    case_df : pandas DataFrame
        Post-processed DataFrame with all cases
    sigtor_df : pandas DataFrame
        Post-processed DatFrame with only nontornadic and significantly tornadic cases

    """
    
    case_df = pd.read_excel(fname)
        
    # Drop rows based on year
    
    for y in drop_yrs:
        case_df.drop(np.where(case_df['year'] == y)[0], axis=0, inplace=True)
    
    # Create dataset that only contains significantly tornadic and null events
    
    sigtor_df = case_df[np.logical_not(np.logical_and(case_df['type'] == 'TORNADO',
                                                      case_df['magnitude'] <= 1))]
    
    # Reset indices
    
    case_df.reset_index(inplace=True)
    sigtor_df.reset_index(inplace=True)
    
    return case_df, sigtor_df


def pull_model_data(time, forecastHour=0, storage=''):
    """
    Pulls RAP/RUC data off the NCEI server.
    
    Parameters
    ----------
        time : datetime object
            RAP/RUC start time
        forecastHour : integer, optional
            Forecast hour to pull from NCEI. Current options on NCEI are 0 or 1 (default is 0)
        storage : string, optional 
            Local location to store RAP/RUC data pulled down from NCEI (default is '')
            
    Returns
    -------
        file : string 
            Name of RAP/RUC file
    
    Notes
    -----
    Original author: Dylan Steinkruger (Penn State)
    
    Docs: https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/rapid-refresh-rap
    
    - RUC becomes RAP on 20120508
    - 13-km RUC analyses (ruc2anl_130) are not available until 20081029 at 2300 UTC
    - 20-km RUC/RAP analyses (ruc2anl_252 or rap_252) are available from 20050101 to present day
    - Some RUC analysis files have multiple time steps, which makes them larger (~40 MB) and can
        also cause errors during grib parsing
    
    """
    
    # RUC becomes RAP on May 8, 2012
    if time < datetime.datetime(2012, 5, 8):
        file = time.strftime('ruc2anl_252_%Y%m%d_%H00_000.grb')
    else:
        file = time.strftime('rap_252_%Y%m%d_%H00_000.grb2') 
    parent = 'https://www.ncei.noaa.gov/data/rapid-refresh/access/historical/analysis/%Y%m/%Y%m%d/'
    directory = time.strftime(parent)
    
    my_file = Path(storage + file)
    if my_file.is_file():
        print("File Already Downloaded")
        return file
    
    r = requests.get(directory + file, allow_redirects=True)
    open(storage + file, 'wb').write(r.content)
    return file


"""
End spc_database.py
"""
