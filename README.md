# Data Generation for Stability Studies of Power Systems with High Penetration of Inverter-Based Resources

This repository provides tools to generate datasets for small-signal stability analysis using the method proposed in [1]. 

## Guide to set up the data generation tool

Clone stability analysis repository, custom GridCal repository and datagen
```bash
git clone https://github.com/iraola/stability-analysis stability_analysis
git clone https://github.com/iraola/new-GridCal.git GridCal
git clone https://github.com/MauroGarciaLorenzo/datagen
```

Move into datagen and checkout to the necessary branch and create "packages" dir
```bash
cd datagen
git checkout store-dataframes-on-disk
mkdir packages
```

Load python
```bash
module load python/3.12.1
```

Install packages separately into the "packages" directory to make sure we get the library versions we want instead of the ones imposed by GridCal
```bash
pip install -r requirements.txt --target=packages/
pip install -e ../GridCal/src/GridCalEngine
pip install -e ../stability_analysis
```
Run the code run_datagen_ACOPF.py importing the .yaml file with the desired setup.

#### Preferred configuration for distributed performance on HPC

- Use `@constraint(local=True)` for the main agent task
- Tipology `tree` instead of `plain`
- Use scheduler `orderstrict.FIFOts`

## Acknowledgment
This work has been carried out within the project TED2021-130351B-C21 (HP2C-DT), funded by MICIU/AEI/10.13039/501100011033 and by the European Union NextGenerationEU/PRTR.

## References
[1] Rossi, F., Lorenzo, M. G., de Acevedo, E. I., Barriendos, E. M., Lacerda, V. A., Lordan-Gomis, F., Badia, Rosa & Prieto-Araujo, E. (2025). Data Generation for Stability Studies of Power Systems with High Penetration of Inverter-Based Resources. arXiv preprint arXiv:2512.06369.
