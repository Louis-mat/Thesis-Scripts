# # Generally useful functions on COMPAS output

import h5py as h5
import numpy as np
import pandas as pd



def printCompasDetails(data, *seeds, mask=()):
    """
    Function to print the full Compas output for given seeds, optionally with an additional mask
    """
    list_of_keys = list(data.keys())

    # Check if seed parameter exists - if not, just print without (e.g RunDetails)
    if ('SEED' in list_of_keys) | ('SEED>MT' in list_of_keys): # Most output files

        # Set the seed name parameter, mask on seeds as needed, and set the index
        seedVariableName='SEED' if ('SEED' in list_of_keys) else 'SEED>MT'
        list_of_keys.remove(seedVariableName) # this is the index above, don't want to include it
    
        allSeeds = data[seedVariableName][()]
        seedsMask = np.in1d(allSeeds, seeds)
        if len(seeds) == 0: # if any seed is included, do not reset the mask
            seedsMask = np.ones_like(allSeeds).astype(bool)
        if mask is ():
            mask = np.ones_like(allSeeds).astype(bool)
        mask &= seedsMask

        df = pd.DataFrame.from_dict({param: data[param][()][mask] for param in list(data.keys())}).set_index(seedVariableName).T

    else: # No seed parameter, so do custom print for Run Details

        # Get just the keys without the -Derivation suffix - those will be a second column
        keys_not_derivations = []
        for key in list_of_keys:
            if '-Derivation' not in key:
                keys_not_derivations.append(key)
        
        # Some parameter values are string types, formatted as np.bytes_, need to convert back
        def convert_strings(param_array):
            if isinstance(param_array[0], np.bytes_):
                return param_array.astype(str)
            else:
                return param_array

        df_keys = pd.DataFrame.from_dict({param: convert_strings(data[param][()]) for param in keys_not_derivations }).T
        nCols = df_keys.shape[1] # Required only because if we combine RDs, we get many columns (should fix later)
        df_keys.columns = ['Parameter']*nCols
        df_drvs = pd.DataFrame.from_dict({param: convert_strings(data[param+'-Derivation'][()]) for param in keys_not_derivations }).T
        df_drvs.columns = ['Derivation']*nCols
        df = pd.concat([df_keys, df_drvs], axis=1)

    # Add units as first col
    units_dict = dict()
    for key in list_of_keys:
        val = data[key].attrs['units'].astype(str) 
        if (val == 'Event') or (val == 'State'):   # Swap out Event and State for Bool, it's somewhat clearer here
            val = 'Bool'
        units_dict.update({key:val})
    df.insert(loc=0, column='(units)', value=pd.Series(units_dict))
    return df





def generateGridAndArgsFilesForSeeds(myf=None, fname=None, seeds=None, fname_grid='recreated_grid.txt', fname_args='recreated_args.txt'):
    """
    Create the grid file (and args file, if necessary) to reproduce a set of seeds from a simulation.
    IN:
        myf: a COMPAS hdf5 file (must include either this or fname, not both)
        fname: the pathname to a COMPAS.h5 file (must include either this or myf, not both)
        seeds: the set of seeds to print out
        fname_grid: the name of the gridfile
        fname_args: the name of the argsfile
    """
    
    if (myf == None) == (fname == None):
        print("Need to specify one of myf or fname")
        return
    if (fname is not None):
        myf = h5.File(fname, 'r')
    if (seeds is None) or (len(seeds) < 1):
        print("Need to specify seeds")
        return
        
    RDs = myf['Run_Details']
    SPs = myf['BSE_System_Parameters']

    
    ### Extract the RDs args which are user supplied
    # Get value and derivation columns, regardless of how many are there in total
    df = printCompasDetails(RDs).iloc[:, [1,-1]] 
    # Get a dictionary mapping USER_SUPPLIED input args to values 
    args_dict = df[df['Derivation'] == 'USER_SUPPLIED'].iloc[:, 0].to_dict()
    # Get rid of args which are always irrelevant
    for key in ['grid', 'output-path', 'output-container', 'random-seed', 'number-of-systems']:
        args_dict.pop(key, None)

    spSeeds = SPs['SEED'][()].astype(int)
    mask = np.in1d(SPs['SEED'][()], seeds)

    ### Determine which parameters should go in the grid file 
    dictParamsOfInterest = {
            'initial-mass-1': SPs['Mass@ZAMS(1)'][()][mask],
            'initial-mass-2': SPs['Mass@ZAMS(2)'][()][mask],
            'semi-major-axis': SPs['SemiMajorAxis@ZAMS'][()][mask],
    }

    
    ### Define the parameters which should be checked in RDs, and if included, should be added to the grid
    # Parameters which do not apply to each star
    dictRdKeysToSpKeys = {
            'eccentricity' : 'Eccentricity@ZAMS',
    }
    # Parameters which apply to one star or the other
    dictRdKeysToSpKeysPerStar = {
        'kick-magnitude':        'Kick_Magnitude',
        'kick-magnitude-random': 'Kick_Magnitude_Random',
        'kick-mean-anomaly':     'Kick_Mean_Anomaly',
        'kick-phi':              'Kick_Phi',
        'kick-theta':            'Kick_Theta',
    }
    # Add these in with the appropriate star number
    dictRdKeysToSpKeys.update({ '{}-{}'.format(rdkey, which): '{}({})'.format(spkey, which) for which in [1, 2] for rdkey, spkey in dictRdKeysToSpKeysPerStar.items()})
    for rdkey, spkey in dictRdKeysToSpKeys.items():
        if rdkey in args_dict: # param was set manually
            dictParamsOfInterest.update({'--{}'.format(rdkey): SPs[spkey][()][mask]})
            del args_dict[rdkey] # remove from the args dict so it is not included in the args file
    
    
    ### Write the grid file, removing RDs args as needed
    with open(fname_grid, 'w') as fwrite:
        for ss, seed in enumerate(spSeeds[mask]):
            args_str = '--random-seed {} '.format(seed)
            args_str += ' '.join(['--{} {}'.format(key, val[ss]) for key, val in dictParamsOfInterest.items()]) + '\n'
            fwrite.write(args_str)

            
    ### Write the recreated_args.txt file
    if len(args_dict) > 0:
        with open(fname_args, 'w') as fwrite:
            args_str = ' '.join(['--{} {}'.format(key, val) for key, val in args_dict.items()])
            fwrite.write(args_str)
            
    print("Success!")
    print("Grid file: ", fname_grid)
    if len(args_dict) > 0:
        print("Args file: ", fname_args)
    else:
        print("No Args file produced")





def bootstrapCompas(nHits=None, rate=None, nTotalCompasBinaries=1e6, nIterations=100, nMsolPerCompasBinary=90): 
    """
    Function to calculate the yield and uncertainty per Msol SFR of 
    a given object of interest in COMPAS
    INPUT:
        nHits: integer, number of occurences of the object of interest
        rate: float, COMPAS rate (nHits/nTotalCompasBinaries) - either this or nHits is required
        nTotalCompasBinaries: integer, total number of binaries evolved
        nIterations: number of redraws to perform
        nMsolPerCompasBinary: float, number of solar masses formed per compas binary (fiducial value is 90)
    OUTPUT:
        ratePerMsol: rate of occurence of binary per Msol
        sigPerMsol: uncertainty on that rate
    """

    # Check the input
    if (nHits == None) == (rate == None):
        raise Exception('Need to input only one of nHits or rate')
    elif (rate != None):
        nHits = rate*nTotalCompasBinaries


    # Calculate rate and uncertainty on the hits
    nHits = int(nHits)
    nTotalCompasBinaries = int(nTotalCompasBinaries)
    perMsol = 1/nTotalCompasBinaries/nMsolPerCompasBinary  # 90 is approximation to the number of Msol per compas binary

    ratePerMsol = nHits*perMsol 

    resampledHits = np.sum(np.random.rand(nTotalCompasBinaries, nIterations) < (nHits/nTotalCompasBinaries), axis=0) 
    sigPerMsol = np.std(resampledHits)*perMsol

    return ratePerMsol, sigPerMsol

