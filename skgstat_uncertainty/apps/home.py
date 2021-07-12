import streamlit as st

from skgstat_uncertainty.core import Project
# from ..core import Project

def st_app(project: Project):
    # st.title('Home')

    st.title('SciKit-GStat uncertainty extension')

    st.markdown("""
    This application is an extension to estimate, propagate and analyze uncertainties 
    for variogram estimation using SciKit-GStat.

    You can start off with creating a new Project from the OPTIONS menu above, select an 
    existing project or upload local saved Projects.
    There are various analysis and propagation tools available, which are organized 
    into chapters. Currently, the following chapters are available:

    * Chapter 1: Variogram estimation and uncertainty margin base data
    * Chapter 2: Theoretical variogram fitting with respect to uncertainty margins
    * Chapter 3: Model influence in kriging application

    ### SciKit-GStat on Github

    The variogram estimation is done with SciKit-GStat. Kriging is done using `gstools`.
    Don't forget to check out `scikit-gstat` and `gstools` on Github:
    """)
    left, right = st.beta_columns(2)
    left.markdown("[![scikit-gstat on Github](https://github-readme-stats.vercel.app/api/pin?username=mmaelicke&repo=scikit-gstat&theme=dark)](https://github.com/mmaelicke/scikit-gstat)")
    right.markdown("[![gstools on Github](https://github-readme-stats.vercel.app/api/pin?username=geostat-framework&repo=gstools&theme=dark)](https://github.com/geostat-framework/gstools)")

    st.write("""
    <div style="display: flex; flex-direction: row; margin-top: 10rem;"><a href="https://hydrocode.de" target="_blank">
    <img height="60" width="60" src="https://firebasestorage.googleapis.com/v0/b/hydrocode-website.appspot.com/o/public%2Flogo.png?alt=media&token=8dda885c-0a7d-4d66-b5f6-072ddabf3b02">
    <span style="font-size: 220%; margin-left: 1.5rem;"> a hydrocode application</span>
    </a></div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    st_app()