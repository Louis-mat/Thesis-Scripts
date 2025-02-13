# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

# %matplotlib inline

# for testing
path = 'COMPAS_Output.h5' 
myf = h5.File(path, 'r')
myf.keys()
MTs = myf['BSE_RLOF']
#printCompasDetails(MTs)



# +
## Defining the stellar types

MS = [0,1,16] # includes CHE stars
FGB = [3]
HG = [4]
CHeB = [5, 6]
preHeMS = MS + FGB + HG + CHeB
HeMS = [7]
HeWD = [10]
WDs = [10,11,12]

# +
# switching between with units

kg = ( 1.988416 * 10**(30) )**(-1) # in solar mass
g = ( 1.988416 * 10**(33) )**(-1) # in solar mass

m = ( 6.957 * 10**(8) )**(-1) # in solar radii
cm = ( 6.957 * 10**(10) )**(-1) # in solar radii

s = ( 86400 )**(-1) # in days

G =  6.67430 * 10**(-11) * kg**(-1) * s**(-2) * m**(3) # in (solar radii)^3 (solar masses)^(-1) days^(-2)
G_cgs = 6.67430 * 10**(-8) # in cm^3 g^-1 s^-2


# +
def getZ(path):
    return h5.File(path, 'r')['BSE_System_Parameters']['Metallicity@ZAMS(1)'][()][0]

def getLogg(mass, radius_rsol):
    m_cgs = mass/g
    r_cgs = radius_rsol/cm
    logg = np.log10( G_cgs * ( m_cgs / r_cgs**2) )
    return logg


# +
# Kiel checks

def checkKiel(logT, logg):
    
    checks = np.zeros_like(logT, dtype=bool)
    
    logT_94 = logT * 9.4

    ## Run through 3 different mask regimes
    # Check 1
    mask = (39.7 <= logT_94) & (logT94 < 41.7)
    checks[mask] = (logg[mask] > -3.4*logT[mask] + 19.3) & (logg[mask] < 6*logT[mask] - 20.4)
    # Check 2
    mask = (41.7 <= logT_94) & (logT94 < 42.2)
    checks[mask] = (logg[mask] > 6*logT[mask] - 22.4) & (logg[mask] < 6*logT[mask] - 20.4)
    # Check 3
    mask = (42.2 <= logT_94) & (logT94 < 44.2)
    checks[mask] = (logg[mask] > 6*logT[mask] - 22.4) & (logg[mask] < -3.4*logT[mask] + 21.8)
    # TODO: what about outside this range? - this is probably just the specified box... Must be a parallelogram-ish
        
    return checks  

def applyKielSelection(path, mask, primary_or_secondary):

    data = h5.File(path)
    RLOF = data['BSE_RLOF']
    
    logT1 = np.log10(RLOF['Teff(1)'][()])
    logT2 = np.log10(RLOF['Teff(2)'][()])
    m1 = RLOF['Mass(1)>MT'][()] 
    m2 = RLOF['Mass(2)>MT'][()] 
    r1 = RLOF['Radius(1)>MT'][()] 
    r2 = RLOF['Radius(2)>MT'][()] 
    logg1 = getLogg(m1, r1)
    logg2 = getLogg(m2, r2)

    mask_Kiel1 = checkKiel(logT1, logg1)
    mask_Kiel2 = checkKiel(logT2, logg2)

    return mask_Kiel1, mask_Kiel2



# +
# Nicolas code
def getPQandD(M, MHeF, logZ):

    # This follows from the equations after Eq. 38 in Hurley+ 2000

    # only mass is an array
    P_arr = np.zeros_like(M)
    Q_arr = np.zeros_like(M)
    logD_arr = np.zeros_like(M)

    Plo = 6
    Phi = 5
    Qlo = 3
    Qhi = 2
    D0 = 5.37 + 0.135 * logZ
    Dlo = D0
    def D_hi(m):
        D1 = 0.975 * D0 - 0.18 * m
        D2 = 0.5 * D0 - 0.06 * m
        return np.maximum(-1, np.maximum(D1, D2))

    mask_m_lt_mhef = M <= MHeF
    mask_m_gtr_2p5 = M > 2.5

    # Case 1: M <= MHeF
    mask = mask_m_lt_mhef

    P_arr[mask]    = Plo
    Q_arr[mask]    = Qlo
    logD_arr[mask] = Dlo

    # Case 2: M > 2.5
    mask = ~mask_m_lt_mhef & mask_m_gtr_2p5

    P_arr[mask]    = Phi
    Q_arr[mask]    = Qhi
    logD_arr[mask] = D_hi(M[mask])

    # Case 3: MHeF < M <= 2.5
    mask = ~mask_m_lt_mhef & ~mask_m_gtr_2p5

    # interpolate linearly in M between MHeF and 2.5
    P_arr[mask] = np.interp(M[mask], [MHeF, 2.5], [Plo, Phi])
    Q_arr[mask] = np.interp(M[mask], [MHeF, 2.5], [Qlo, Qhi])
    logD_arr[mask] = np.interp(M[mask], [MHeF, 2.5], [Dlo, D_hi(2.5)])

    D_arr = np.power(10, logD_arr)
    return P_arr, Q_arr, D_arr

def CoreMassFromLuminosity(Lx, B, D, q, p, Lref):
    mCore = np.zeros_like(Lx)
    # Case 1
    mask = Lref > Lx
    mCore[mask] = np.power(Lref/B, 1/q)[mask]
    # Case 2
    mask = ~mask
    mCore[mask] = np.power(Lref/D, 1/p)[mask]
    return mCore

def getMHeF(logZ):
    # logZ = np.log10(metallicity/0.02)
    return 1.995 + 0.25*logZ + 0.087*logZ*logZ #Hurley 2000 eq 2

def getCoreMassAtHeIgnition(M, Z):
    # functions of Z
    logZ = np.log10(Z/0.02)
    logZ2 = logZ * logZ
    logZ3 = logZ2 * logZ
    MHeF = getMHeF(logZ)
    p, q, D = getPQandD(M, MHeF, logZ)
    B = np.maximum(30000, 500 + (17500 * np.power(M,0.6))) 
    Mx = np.power(B/D, 1/(p-q))
    Lx = np.minimum(B*np.power(Mx,q), D*np.power(Mx, p)) 
    # crossing mass and luminosity, arrays as functions of the total mass
    # constants
    b9 = 2751.631 + 355.7098 * logZ
    b10 = -0.03820831 + 0.05872664 * logZ
    b11 = 1.071738E2 - 8.970339E1 * logZ - 3.949739E1 * logZ2
    b11 = b11 * b11
    b12 = 7.348793E2 - 1.531020E2 * logZ - 3.7937E1 * logZ2
    b13 = 9.219293 - 2.005865 * logZ - 5.561309 * logZ2 / 10
    b13 = b13 * b13
    b36 = 0.1445216 - 0.06180219* logZ + 0.03093878 * logZ2 + 0.01567090 * logZ3
    b36 = b36 * b36 * b36 * b36
    b37 = 1.304129 + 0.1395919 * logZ + 0.004142455 * logZ2 - 0.009732503 * logZ3
    b37 = 4 * b37
    b38 = 0.5114149 - 0.01160850 * logZ
    b38 = b38 * b38 * b38 * b38
    # physical parameters
    L_MHeF = (b11 + (b12 * np.power(MHeF, 3.8))) / (b13 + MHeF * MHeF)
    Mc_MHeF = CoreMassFromLuminosity(Lx, B, D, q, p, L_MHeF) 
    McBAGB = b36 * np.power(M, b37) + b38
    C1 = 9.20925E-5
    C2 = 5.402216
    alpha1 = ((b9 * np.power(M, b10)) - L_MHeF) / L_MHeF
    LHeI = (b9 * np.power(M, b10)) / (1 + (alpha1 * np.exp(15 * (M - MHeF))))
    C = np.power(Mc_MHeF, 4) - (C1 * np.power(MHeF, C2))
    # Combine to get core mass at He
    mask = M > MHeF
    MCoreAtHeI_lo = CoreMassFromLuminosity(Lx, B, D, q, p, LHeI) 
    MCoreAtHeI_hi = np.minimum(0.95 * McBAGB, np.sqrt(np.sqrt(C + C1 * np.power(M, C2)))) 
    MCoreAtHeI = MCoreAtHeI_lo
    MCoreAtHeI[mask] = MCoreAtHeI_hi[mask]
    return MCoreAtHeI

def getMasks3pc5pcMassIgnition(massPre, massHe, Z):
    MCoreAtHeI = getCoreMassAtHeIgnition(massPre, Z)
    mask3pc = massHe >= (MCoreAtHeI * 0.97)  # 3%
    mask5pc = massHe >= (MCoreAtHeI * 0.95)  # 5%
    return(MCoreAtHeI, mask3pc, mask5pc)

massPre = np.linspace(0.5, 5, 100)
massHe = 0.6 * massPre + 0.1*np.random.rand(massPre.shape[0])-0.05 #add some scatter
Z = 0.02

getMasks3pc5pcMassIgnition(massPre, massHe, Z)

# Usage
# TODO RTW: do we need to worry about including things that do not climb the RGB?
#core_mass_at_RGB_tip, mask3pc, mask5pc = getMasks3pc5pcMassIgnition(progenitor_mass, wd_mass, corresponding_metallicities)

# +
# SdB means a HeMS star with mass < 0.8

def get_all_SdB_masks(path):
    data = h5.File(path)
    RLOF = data['BSE_RLOF']

    # All masses
    mass1_prev = RLOF['Mass(1)<MT'][()]
    mass2_prev = RLOF['Mass(2)<MT'][()]
    mass1_post = RLOF['Mass(1)>MT'][()]
    mass2_post = RLOF['Mass(2)>MT'][()]

    # All stellar types
    stype1_prev = RLOF['Stellar_Type(1)<MT'][()]
    stype2_prev = RLOF['Stellar_Type(2)<MT'][()]
    stype1_post = RLOF['Stellar_Type(1)>MT'][()]
    stype2_post = RLOF['Stellar_Type(2)>MT'][()]

    # CEE or stable RLOF, non merger
    is_ce = RLOF['CEE>MT'][()]==1
    not_merged = RLOF['Merger'][()]==0
    #is_stable = ~is_ce

    # was donor in MT
    is_donor1 = RLOF['RLOF(1)>MT'][()]==1
    is_donor2 = RLOF['RLOF(2)>MT'][()]==1

    # Combine to masks

    # confirmed SdBs
    is_sdB1 = np.in1d(stype1_prev, preHeMS) & np.in1d(stype1_post, HeMS) & (mass1_post <= 0.8)
    is_sdB2 = np.in1d(stype2_prev, preHeMS) & np.in1d(stype2_post, HeMS) & (mass2_post <= 0.8)

    # candidate WDs
    is_WD1 =  np.in1d(stype1_prev, preHeMS) & np.in1d(stype1_post, HeWD) & (mass1_post <= 0.8)
    is_WD2 =  np.in1d(stype2_prev, preHeMS) & np.in1d(stype2_post, HeWD) & (mass2_post <= 0.8)

    # Identify which candidate WDs pass the 3% or 5% thresholds
    Z = getZ(path)
    _, mask_mCore1_within_3pc, mask_mCore1_within_5pc = getMasks3pc5pcMassIgnition(mass1_prev, mass1_post, Z)
    _, mask_mCore2_within_3pc, mask_mCore2_within_5pc = getMasks3pc5pcMassIgnition(mass2_prev, mass2_post, Z)
    is_missing_sdB1_3pc = is_WD1 & mask_mCore1_within_3pc
    is_missing_sdB1_5pc = is_WD1 & mask_mCore1_within_5pc
    is_missing_sdB2_3pc = is_WD2 & mask_mCore2_within_3pc
    is_missing_sdB2_5pc = is_WD2 & mask_mCore2_within_5pc

    return is_ce, not_merged, is_sdB1, is_sdB2, is_missing_sdB1_3pc, is_missing_sdB1_5pc, is_missing_sdB2_3pc, is_missing_sdB2_5pc
# -









