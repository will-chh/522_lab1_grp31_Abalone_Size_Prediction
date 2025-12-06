# use the miniforge base, make sure you specify a verion
FROM condaforge/miniforge3:latest

# copy the lockfile into the container
COPY conda-lock.yml conda-lock.yml

# setup conda-lock
RUN conda install -n base -c conda-forge conda-lock -y

# install packages from lockfile into dockerlock environment
RUN conda-lock install -n dockerlock conda-lock.yml

# make dockerlock the default environment
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate dockerlock" >> ~/.bashrc

# set the default shell to use bash with login to pick up bashrc
# this ensures that we are starting from an activated dockerlock environment
SHELL ["/bin/bash", "-l", "-c"]

# expose JupyterLab port
EXPOSE 8888

# sets the default working directory
# this is also specified in the compose file
WORKDIR /workplace

# run JupyterLab on container start
# uses the jupyterlab from the install environment
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "dockerlock" ]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=''", "--ServerApp.password=''"]