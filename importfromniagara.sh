#!/bin/bash

SERVER="vasissua@niagara.scinet.utoronto.ca"
SCRATCH="/scratch/m/murray/vasissua"
SERVER_FOLDER="/Zeldovich3/output_23.04.24:1/*.hdf5"

LOCAL_FOLDER="../output_23.04.24:1/"

rsync "$SERVER:""$SCRATCH""$SERVER_FOLDER" "$LCOAL_FOLDER"

