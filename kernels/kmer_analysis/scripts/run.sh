#/bin/bash
BENCHMARK=$1
HOST1=$2
HOST2=$3
HOST3=$4
HOST4=$5

DEVICE_ID_1=0
DEVICE_ID_2=1
DEVICE_ID_3=2
DEVICE_ID_4=3

HASH=`date|md5sum|head -c 5`
FILENAME="${BENCHMARK}_${HASH}"
echo $FILENAME
#VARS0="--bind-to core --map-by socket"
VARS1="-x LD_LIBRARY_PATH=/projects/ppc64le-pwr9-rhel8/tpls/cuda/12.0.0/gcc/12.2.0/base/rantbbm/lib64/:$LD_LIBRARY_PATH"

# # #1 node
# FILENAME_ACTUAL=$FILENAME"_1x1xX.res"
# echo "time" | tee $FILENAME_ACTUAL 
# let NP=2
# for i in $(seq 1 7); do 
#    let NPN=$NP/1
#    for reps in $(seq 1 3); do
#       GASNET_PHYSMEM_MAX='207 GB' CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 mpirun -np $NPN -npernode $NPN $VARS0 $VARS1 -host $HOST1:$NP  ./$BENCHMARK ../Athal_mito.fas.txt 1000000 100 31 | tee -a $FILENAME_ACTUAL
#    done 
#    let NP=$NP*2   
# done

#2 nodes

# #1 node
FILENAME_ACTUAL=$FILENAME"_1x1xX.res"
echo "time" | tee $FILENAME_ACTUAL 
let NP=2
for i in $(seq 1 1); do 
   let NPN=$NP/2
   for reps in $(seq 1 1); do
      #GASNET_PHYSMEM_MAX='207 GB' CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 upcxx-run -np $NP -npernode $NPN $VARS0 $VARS1 -host $HOST1:$NPN,$HOST2:$NPN  ./$BENCHMARK ../Athal_mito.fas.txt 1000000 100 31 | tee -a $FILENAME_ACTUAL
      GASNET_PHYSMEM_MAX='207 GB' CUDA_VISIBLE_DEVICES=$DEVICE_ID_1 upcxx-run -v -np $NP -N 2 -ssh-servers $HOST1,$HOST2  ./$BENCHMARK ../Athal_mito.fas.txt 1000000 100 31 | tee -a $FILENAME_ACTUAL
   done 
   let NP=$NP*2   
done

