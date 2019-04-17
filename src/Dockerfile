# Dockerfile with software for 
# Image Analysis, Classification and Change Detection
# in Remote Sensing, Fourth Revised Edition
 
FROM     debian:stretch  
           
MAINTAINER Mort Canty "mort.canty@gmail.com"
ENV     REFRESHED_AT 2018-05-17

RUN apt-get update && apt-get install -y \
    python \
    build-essential \
    make \
    gcc \
    pandoc \
    python-dev \
    python-pygments \
    python-pip \
    git 

# ipython notebook
RUN     pip install --upgrade pip
RUN     pip install jupyter==1.0.0
RUN     pip install ipyleaflet==0.9.0
RUN     jupyter nbextension enable --py --sys-prefix ipyleaflet

# tensorflow
RUN     pip install tensorflow

# enable parallel computing
RUN     pip install ipyparallel
RUN     jupyter notebook --generate-config
RUN     sed -i "/# Configuration file for jupyter-notebook./a  \ 
                c.NotebookApp.server_extensions.append('ipyparallel.nbextension')" \
            /root/.jupyter/jupyter_notebook_config.py

# install python environment for crc4 scripts
RUN     apt-get install -y python-numpy python-scipy \
         python-gdal  libgdal-dev gdal-bin python-shapely \
         python-opencv
         
RUN     pip install --upgrade matplotlib    

# install mlpy (with MaximumLikelihoodC and LibSvm)
RUN     apt-get install -y libgsl0-dev
ADD     mlpy-3.5.0 /mlpy-3.5.0
WORKDIR /mlpy-3.5.0
RUN     python setup.py install

# setup the prov_means library
COPY    prov_means.c /home/prov_means.c
WORKDIR /home
RUN     gcc -shared -Wall -g -o libprov_means.so -fPIC prov_means.c
RUN     cp libprov_means.so /usr/lib/libprov_means.so
RUN     rm prov_means.c

EXPOSE 8888

# setup for earthengine
RUN     pip install pyasn1 --upgrade
RUN     pip install --upgrade setuptools && \
        pip install google-api-python-client && \
        pip install --upgrade oauth2client && \
        pip install pyCrypto && \
        apt-get install -y libssl-dev
RUN     pip install earthengine-api

# opencv
RUN     pip install opencv-python

# install auxil
COPY    dist/auxil-1.0.tar.gz /home/auxil-1.0.tar.gz
RUN     tar -xzvf auxil-1.0.tar.gz
WORKDIR /home/auxil-1.0
RUN     python setup.py install  
WORKDIR /home
RUN     rm -rf auxil-1.0
RUN     rm auxil-1.0.tar.gz

# SPy
RUN     pip install spectral

# textbook scripts, notebooks and images
ADD     scripts /home/scripts
ADD     auxil /home/auxil 
ADD     imagery_initial /home/imagery
COPY    Chapter1.ipynb /home/Chapter1.ipynb
COPY    Chapter2.ipynb /home/Chapter2.ipynb
COPY    Chapter3.ipynb /home/Chapter3.ipynb
COPY    Chapter4.ipynb /home/Chapter4.ipynb
COPY    Chapter5_1.ipynb /home/Chapter5_1.ipynb
COPY    Chapter5_2.ipynb /home/Chapter5_2.ipynb
COPY    Chapter6.ipynb /home/Chapter6.ipynb
COPY    Chapter7.ipynb /home/Chapter7.ipynb
COPY    Chapter8.ipynb /home/Chapter8.ipynb
COPY    Chapter9.ipynb /home/Chapter9.ipynb

# ipython notebook startup script
ADD     notebook.sh /
RUN     chmod u+x /notebook.sh

WORKDIR /home  
CMD     ["/notebook.sh"]
