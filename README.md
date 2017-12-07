CRCDocker
=========
Source files for the Docker image mort/crcdocker

Command line versions of several Python scripts for the textbook "Image Analysis, Classification and Change Detection in Remote Sensing"

On Ubuntu, for example, pull and run the container with

sudo docker run -d -p 433:8888 -v my_images:/crc/imagery/ �name=crc mort/crcdocker

This maps the host directory my_images to the container directory /crc/imagery/ and runs the
container as a daemon which is serving iPython notebooks. 

Point your browser to http://localhost:433 to see the iPython notebook home page. 

Open a new notebook and see the available scripts with

In[1] cd /crc

In[2] ls -l
