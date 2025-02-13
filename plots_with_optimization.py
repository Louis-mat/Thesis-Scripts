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

# %matplotlib inline

from optimized_scripts import get_all_SdB_masks

path = 'COMPAS_Output.h5' 


# +
def make_plot(ax, path, bins=None, include_primaries=True, include_secondaries=True, include_3pc=False, include_5pc=False):
    data = h5.File(path)
    RLOF = data['BSE_RLOF']

    if include_3pc and include_5pc:
        raise Exception("only include 3 or 5")

    # All masses
    mass1_prev = RLOF['Mass(1)<MT'][()]
    mass2_prev = RLOF['Mass(2)<MT'][()]
    mass1_post = RLOF['Mass(1)>MT'][()]
    mass2_post = RLOF['Mass(2)>MT'][()]

    is_ce, not_merged, is_sdB1, is_sdB2, is_missing_sdB1_3pc, is_missing_sdB1_5pc, is_missing_sdB2_3pc, is_missing_sdB2_5pc = get_all_SdB_masks(path)
    
    all_masses = []
    all_labels = []
        
    if include_primaries:
        mass_sdB1_cee = mass1_post[is_sdB1 & is_ce & not_merged]
        mass_sdB1_smt = mass1_post[is_sdB1 & ~is_ce & not_merged]
        all_masses.append(mass_sdB1_cee)
        all_masses.append(mass_sdB1_smt)
        all_labels.append( "SdB 1 CEE" )
        all_labels.append( "SdB 1 SMT" )
    
    if include_secondaries:
        mass_sdB2_cee = mass2_post[is_sdB2 & is_ce & not_merged]
        mass_sdB2_smt = mass2_post[is_sdB2 & ~is_ce & not_merged]
        all_masses.append(mass_sdB2_cee)
        all_masses.append(mass_sdB2_smt)
        all_labels.append( "SdB 2 CEE" )
        all_labels.append( "SdB 2 SMT" )
    
    if include_3pc:
        if include_primaries:
            mass_missing_sdB1_3pc_cee = mass1_post[is_missing_sdB1_3pc & is_ce & not_merged]
            mass_missing_sdB1_3pc_smt = mass1_post[is_missing_sdB1_3pc & ~is_ce & not_merged]
            all_masses.append(mass_missing_sdB1_3pc_cee)
            all_masses.append(mass_missing_sdB1_3pc_smt)
            all_labels.append( "3% WD 1 CEE" )
            all_labels.append( "3% WD 1 SMT" )

        if include_secondaries:
            mass_missing_sdB2_3pc_cee = mass2_post[is_missing_sdB2_3pc & is_ce & not_merged]
            mass_missing_sdB2_3pc_smt = mass2_post[is_missing_sdB2_3pc & ~is_ce & not_merged]
            all_masses.append(mass_missing_sdB2_3pc_cee)
            all_masses.append(mass_missing_sdB2_3pc_smt)
            all_labels.append( "3% WD 2 CEE" )
            all_labels.append( "3% WD 2 SMT" )

    if include_5pc:
        if include_primaries:
            mass_missing_sdB1_5pc_cee = mass1_post[is_missing_sdB1_5pc & is_ce & not_merged]
            mass_missing_sdB1_5pc_smt = mass1_post[is_missing_sdB1_5pc & ~is_ce & not_merged]
            all_masses.append(mass_missing_sdB1_5pc_cee)
            all_masses.append(mass_missing_sdB1_5pc_smt)
            all_labels.append( "5% WD 1 CEE" )
            all_labels.append( "5% WD 1 SMT" )
        
        if include_secondaries:
            mass_missing_sdB2_5pc_cee = mass2_post[is_missing_sdB2_5pc & is_ce & not_merged]
            mass_missing_sdB2_5pc_smt = mass2_post[is_missing_sdB2_5pc & ~is_ce & not_merged]
            all_masses.append(mass_missing_sdB2_5pc_cee)
            all_masses.append(mass_missing_sdB2_5pc_smt)
            all_labels.append( "5% WD 2 CEE" )
            all_labels.append( "5% WD 2 SMT" )

    ax.hist(all_masses, histtype='barstacked', label=all_labels, bins=bins, alpha=0.9, rwidth=0.9, edgecolor='black')

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(15,8))
incs = [ [False, False], [True, False], [False, True] ]
bins = np.linspace(0.2,0.8,18)
for ii, ax in enumerate(axes[0]):
    make_plot(ax, path, bins=bins, include_primaries=True, include_secondaries=False, include_3pc=incs[ii][0], include_5pc=incs[ii][1])
    ax.legend()
    ax.set_ylim(0, 2500)
for ii, ax in enumerate(axes[1]):
    make_plot(ax, path, bins=bins, include_primaries=False, include_secondaries=True, include_3pc=incs[ii][0], include_5pc=incs[ii][1])
    ax.legend()
    ax.set_ylim(0, 500)
# -






