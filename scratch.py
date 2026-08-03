
"""
To do list

1. make a system object containing a potential, coordinate limits, and a diffusion coefficient, a macrostate classifier, and a macrostate free energy difference calculator
2. make a CV object containing a CV, CV gradient, and CV limits
2.5. update tests
2.6. clean up repo
3. write updated WE code
    manually clean and review current code
    feed cleaned code to claude and ask it to check it; also upload huber and kim 1996 
4. make a metadynamics wrapper function matching the WE propagator requirements
5. make simulation system object (and decide if this actually makes sense) w/ macrostate classifier (or make the classifier its own thing with the CV?)
6. write function to set MTD parameters based on the existing one in msm_toy_systems/
    implement binless diffusion coefficient calculation and see if it matches MSM
7. assemble all of this in main and test it


Low priority

100. rewrite test_free_energy_on_cv_grid() in test_propagators_grid.py to use an analytically solvable case instead of another implementation of the same calculation

"""

