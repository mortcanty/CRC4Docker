CRC4Docker
=========
Source files for the Docker image mort/crc4docker

Python scripts for the textbook "Image Analysis, Classification and Change Detection in Remote Sensing, Fourth Revised Edition"

On Ubuntu, for example, pull and/or run the container for the first time with

    sudo docker run -d -p 443:8888 -p 6006:6006 -v <my_image_folder>:/home/myimagery/ --name=crc4 mort/crc4docker

This maps the host directory my_image_folder to the container directory /home/myimagery/ and runs the
container in detached mode. 

Point your browser to http://localhost:443 to see the Jupyter home page and open a notebook.

In order to use Earth Engine scripts,  
you must authenticate. From the home page open a local terminal and enter
    
    earthengine authenticate
    
and follow the instrctions. This adds authentication information to the container and only has to be done once.     

Stop the container with

    sudo docker stop crc4  
     
Re-start with

    sudo docker start crc4     