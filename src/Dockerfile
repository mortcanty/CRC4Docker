# x86 architecture 
FROM     debian:jessie  
           

MAINTAINER Mort Canty "mort.canty@gmail.com"
ENV     REFRESHED_AT 2015-12-26
RUN apt-get update && apt-get install -y \
    python \
    build-essential \
    make \
    gcc \
    pandoc \
    dos2unix \
    python-dev \
    python-pygments \
    python-pip 

# ipython notebook
RUN     pip install "ipython[notebook]"

# enable parallel computing
RUN     pip install ipyparallel
RUN     jupyter notebook --generate-config
RUN     sed -i "/# Configuration file for jupyter-notebook./a  \ 
                c.NotebookApp.server_extensions.append('ipyparallel.nbextension')" \
            /root/.jupyter/jupyter_notebook_config.py

# install python environment for crc scripts
RUN     apt-get install -y python-numpy python-scipy \
         python-gdal python-matplotlib libgdal-dev gdal-bin python-shapely \
         python-opencv

# install *latest* mlpy (with MaximumLikelihoodC and LibSvm)
RUN     apt-get install -y libgsl0-dev
ADD     mlpy-3.5.0 /mlpy-3.5.0
WORKDIR /mlpy-3.5.0
RUN     python setup.py install

# package imported by auxil package (but not used)
RUN     apt-get install -y python-tk 

# setup the prov_means library
COPY    prov_means.c /home/prov_means.c
WORKDIR /home
RUN     gcc -shared -Wall -g -o libprov_means.so -fPIC prov_means.c
RUN     cp libprov_means.so /usr/lib/libprov_means.so

# ipython notebook startup script
ADD     notebook.sh /
RUN     chmod u+x /notebook.sh
RUN     dos2unix /notebook.sh

EXPOSE 8888

# install auxil
ENV     SCRIPTS_CHANGED 2015-12-09
COPY    dist/auxil-1.1.tar.gz /home/auxil-1.1.tar.gz
RUN     tar -xzvf auxil-1.1.tar.gz
WORKDIR /home/auxil-1.1
RUN     python setup.py install  
WORKDIR /home
RUN     rm -rf auxil-1.1
RUN     rm auxil-1.1.tar.gz

# setup for earthengine
RUN     pip install pyasn1 --upgrade
RUN     pip install --upgrade setuptools && \
        pip install google-api-python-client && \
        pip install pyCrypto && \
        apt-get install -y libssl-dev
RUN     pip install earthengine-api
ADD     Crypto /home/Crypto
ADD     ee /home/ee
ADD     httplib2 /home/httplib2
ADD     oauth2client /home/oauth2client

# textbook scripts and images
ADD     Scripts /home/Scripts
ADD     Images  /home/Images
COPY    normalize /home/normalize
COPY    c-correction.sh /home/c-correction.sh

WORKDIR /home  
CMD     ["/notebook.sh"]
