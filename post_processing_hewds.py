# +
# This file contains the old and new versions of getPQandD, and some checks to compare the differences.

# There is a small bug in the old version of the function, in the interpolation of logD between MHeF and 2.5. This is corrected in the new version

# -

import numpy as np


# +
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

    mask_below_mhef = M <= MHeF
    mask_m_gtr_2p5 = M > 2.5

    # Case 1: M <= MHeF
    mask = mask_below_mhef

    P_arr[mask]    = Plo
    Q_arr[mask]    = Qlo
    logD_arr[mask] = Dlo

    # Case 2: M > 2.5
    mask = ~mask_below_mhef & mask_m_gtr_2p5

    P_arr[mask]    = Phi
    Q_arr[mask]    = Qhi
    logD_arr[mask] = D_hi(M[mask])

    # Case 3: MHeF < M <= 2.5
    mask = ~mask_below_mhef & ~mask_m_gtr_2p5

    # interpolate linearly in M between MHeF and 2.5
    P_arr[mask] = np.interp(M[mask], [MHeF, 2.5], [Plo, Phi])
    Q_arr[mask] = np.interp(M[mask], [MHeF, 2.5], [Qlo, Qhi])
    logD_arr[mask] = np.interp(M[mask], [MHeF, 2.5], [Dlo, D_hi(2.5)])

    D_arr = np.power(10, logD_arr)
    return P_arr, Q_arr, D_arr

def CoreMassFromLuminosity(L, B, D, q, p, Lx):
    mCore = np.zeros_like(L)
    # Case 1
    mask = L > Lx
    mCore[mask] = np.power(L/B, 1/q)[mask]
    # Case 2
    mask = ~mask
    mCore[mask] = np.power(L/D, 1/p)[mask]
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
    p, q, D = getPQandD(M, MHeF, logZ)
    B = np.maximum(30000, 500 + (17500 * np.power(M,0.6))) 
    Mx = np.power(B/D, 1/(p-q))
    Lx = np.minimum(B*np.power(Mx,q), D*np.power(Mx, p)) 
    Mc_MHeF = CoreMassFromLuminosity(L_MHeF, B, D, q, p, Lx) 
    McBAGB = b36 * np.power(M, b37) + b38
    C1 = 9.20925E-5
    C2 = 5.402216
    alpha1 = ((b9 * np.power(M, b10)) - L_MHeF) / L_MHeF
    L = (b9 * np.power(M, b10)) / (1 + (alpha1 * np.exp(15 * (M - MHeF))))
    C = np.power(Mc_MHeF, 4) - (C1 * np.power(MHeF, C2))
    # Combine to get core mass at He
    mask = M > MHeF
    MCoreAtHeI_lo = CoreMassFromLuminosity(L, B, D, q, p, Lx) 
    MCoreAtHeI_hi = np.minimum(0.95 * McBAGB, np.sqrt(np.sqrt(C + C1 * np.power(M, C2)))) 
    MCoreAtHeI = MCoreAtHeI_lo
    MCoreAtHeI[mask] = MCoreAtHeI_hi[mask]
    return MCoreAtHeI

def getMasks3pc5pcMassIgnition(massPre, massHe, Z):
    MCoreAtHeI = getCoreMassAtHeIgnition(massPre, Z)
    mask3pc = massHe >= (MCoreAtHeI * 0.97)  # 3%
    mask5pc = massHe >= (MCoreAtHeI * 0.95)  # 5%
    return(MCoreAtHeI, mask3pc, mask5pc)

Usage

# TODO RTW: do we need to worry about including things that do not climb the RGB?
core_mass_at_RGB_tip, mask3pc, mask5pc = maskIgnitesOrNot(progenitor_mass, wd_mass, corresponding_metallicities)
# -



# +

def getPQandD_old(mass, MHeF, logmet):
    outP = []
    outQ = []
    outD = []
    for i in range(len(mass)):
        p = 6
        q = 3
        m = mass[i]
        mhef = MHeF[i]
        lz = logmet[i]
        D0 = 5.37 + 0.135 * lz
        D1 = 0.975 * D0 - 0.18 * m
        logD = D0
        if m > mhef:
            if m >= 2.5:
                p = 5
                q = 2
                D2 = 0.5 * D0 - 0.06 * m
                logD = max(max(-1, D1), D2)
            else:
                gradient = 1 / (mhef - 2.5)
                interceptP = 5 - 2.5 * gradient
                p = gradient * m + interceptP

                interceptQ = 2 - 2.5 * gradient
                q = gradient * m + interceptQ

                gradientQ = gradient * (D0 - D1)
                interceptQ = D0 - mhef * gradientQ
                logD = gradientQ * m + interceptQ

        outP.append(p)
        outQ.append(q)
        outD.append(10**logD)
    return np.array(outP), np.array(outQ), np.array(outD)





# +
mass = np.linspace(0.1, 20, 100)
MHeF = 1.9
logZ = 0

print(mass[:5])

r1 = getPQandD(mass, MHeF, logZ)
r2 = getPQandD_old(mass, MHeF*np.ones_like(mass), logZ*np.ones_like(mass))

d1 = r1[2]
d2 = r2[2]
print(np.array_equal(r1[0],r2[0]))
print(np.array_equal(r1[1],r2[1]))
msk = np.equal(r1[2],r2[2])
print(msk)
print(r1[2][~msk])
print(r2[2][~msk])
print(mass[~msk])
print(msk[7:15])
print()

print(d1[7:15])
print(d2[7:15])
