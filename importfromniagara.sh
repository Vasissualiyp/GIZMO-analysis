#!/bin/bash

SERVER="vasissua@niagara.scinet.utoronto.ca"
SCRATCH="/scratch/m/murray/vasissua"
SERVER_FOLDER="/Zeldovich3/output_23.05.01_512:1/*.hdf5"

LOCAL_FOLDER="../output_23.05.01_512:1/"

rsync --progress "$SERVER:""$SCRATCH""$SERVER_FOLDER" "$LOCAL_FOLDER"

