# MetAnalysis
Thank you for visiting this project! MetAnalysis is an effort to combine all of my Python research code into a central location with tests and examples. As someone who has "stood on the shoulders of giants" when it comes to Python, I encourage anyone to grab pieces of code here that you think will be helpful, so that way you can stand on my shoulders and continue to advance our field.


### Getting Started
The easiest way to use MetAnalysis is to fork this repo or make a clone on your own machine (MetAnalysis is currently not available through other means, like `conda` or `pip`). After making changes, I encourage you to re-run the tests within the tests directory using `pytest`. This should be as easy as switching into the tests directory and running `pytest` from a terminal window. If you are simply looking to use some pieces of code here, Jupyter notebooks containing some usage examples can be found in the examples directory.


### Highlights
* src/idealized_sounding_fcts.py
    - Compute thermodynamic parameters (e.g., CAPE and CIN) using either pseudoadiabatic or reversible parcel ascent
    - Create analytic thermodynamic soundings, such as Weisman and Klemp (1982) and McCaul and Weisman (2001)
    - Includes updated versions of the Python code used in [Murdzek et al. (2021, DOI: 10.1175/JAS-D-21-0069.1)](https://doi.org/10.1175/JAS-D-21-0069.1). examples/mccaul_weisman_snd_parameters.ipynb contains an example of how this code can be used to create the input soundings from Murdzek et al. (2021) and how to compute pseudoadiabatic and reversible parameters.
* src/kine_fcts.py
    - Efficiently compute Eulerian circulation around each point in a 3D grid using Numba
* src/largest_area.py
    - Track supercells using the largest, continuous area where 2-5 km updraft helicity exceeds a threshold


### Contributing
If you find any bugs or have any suggestions for improvements, please open an issue, submit a pull request, or contact me through another channel. I have been known to make mistakes!

Happy coding!
