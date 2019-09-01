# BioNAS-pub
Public version of BioNAS, a Neural Architecture Search method for interpretable Bioinformatics applications


## Install
Clone the GitHub repo, then type the following in your terminal:
```bash
cd BioNAS-pub/
python setup.py install
```
You might have to install [pysam](https://pysam.readthedocs.io/en/latest/api.html) separately. I recommend using conda following instructions [here](https://anaconda.org/bioconda/pysam).

## Usage
Upon sucessfuly installation, you should be able to run BioNAS optimization by
```bash
python examples/mock_conv1d_state_space.py
```

Scripts for reproducing the results in the preprint paper can be found in "analysis/" folder.

Some helper scripts for running BioNAS in SGE can be found in "sge/" folder.

## Contact
Feel free to open a [GitHub Issues](https://github.com/zj-zhang/BioNAS-pub/issues) or contact me at <zj.z@ucla.edu>.