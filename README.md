# skgstat_uncertainty

This repository contains a [Streamlit](https://streamlit.io) applicatoin to analyze uncertainties propagated into experimental and theoretical variograms estimated with [SciKit-GStat](https://github.com/mmaelicke/scikit-gstat) and Kriging applications applied with [GSTools](https://github.com/Geostat-Framework/GSTools).

An online version on streamlit share can be found here: https://share.streamlit.io/hydrocode-de/skgstat_uncertainty/main

There are some space limitations on the online version, which can prevent downloading of result files.

## Local Install

To run the application locally, first clone and install it:

```bash
git clone git@github.com:hydrocode-de/skgstat_uncertainty.git
cd skgstat_uncertainty

pip install -e .
```

Start the application like:

```bash
# this can be run from any working directory
python -m skgstat_uncertainty
```
or
```bash
# this has to be run from the repository root
streamlit run streamlit_app.py
```

