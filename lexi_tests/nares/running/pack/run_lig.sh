#!/bin/bash
cd ~/LIGGGHTS_SEAICE/lexi_tests/nares/simulations/create_packing
mpirun -np 3 /usr/local/bin/liggghts -in in.pack
mpirun -np 3 /usr/local/bin/liggghts -in in.pack_from_restart1
mpirun -np 3 /usr/local/bin/liggghts -in in.pack_from_restart2
mpirun -np 3 /usr/local/bin/liggghts -in in.pack_from_restart3