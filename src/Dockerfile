# Dockerfile with software for 
# Image Analysis, Classification and Change Detection
# in Remote Sensing, Fourth Revised Edition
 
FROM     debian:jessie  
           
MAINTAINER Mort Canty "mort.canty@gmail.com"
ENV     REFRESHED_AT 2018-12-25

RUN apt-get update && apt-get install -y \
    python \
    build-essential \
    make \
    gcc \
    pandoc \
    python-dev \
    python-pygments \
    python-pip \
    libssl-dev \
    git 

RUN     pip install --upgrade pip
COPY    requirements.txt /home
RUN     pip install -r /home/requirements.txt
RUN     rm /home/requirements.txt

RUN apt-get install -y \
    python-gdal \
    libgdal-dev \
    gdal-bin  

RUN     jupyter nbextension enable --py --sys-prefix ipyleaflet

# enable parallel computing
RUN     jupyter notebook --generate-config
RUN     sed -i "/# Configuration file for jupyter-notebook./a  \ 
                c.NotebookApp.server_extensions.append('ipyparallel.nbextension')" \
            /root/.jupyter/jupyter_notebook_config.py
         
# install mlpy (with MaximumLikelihoodC and LibSvm)
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
EXPOSE 6006

# install auxil
COPY    dist/auxil-1.0.tar.gz /home/auxil-1.0.tar.gz
RUN     tar -xzvf auxil-1.0.tar.gz
WORKDIR /home/auxil-1.0
RUN     python setup.py install  
WORKDIR /home
RUN     rm -rf auxil-1.0
RUN     rm auxil-1.0.tar.gz

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
