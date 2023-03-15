As I was using microway server to run the code, could not compile it offline. So here I am adding the instructions to run this code.

Create a folder and copy the histogram_atomic.cu file in that folder. Open terminal and copile the code using nvcc.



nvcc histogram_atomic.cu -o histogram_atomic




An execution file will be created and now we can use this file to run the code.



./histogram_atomic 8 10000




This will print 

1265 1187 1276 1245 1296 1261 1250 1220




While computing throughput we need to compile the code including -G to include debug symbols in the binary.



nvcc -G histogram_atomic.cu -o histogram_atomic



And for performace monitoring 


nvprof ./histogram_atomic 128 100000