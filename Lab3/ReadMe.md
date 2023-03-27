As I was using microway server to run the code, could not compile it offline. So here I am adding the instructions to run this code.

Create a folder and copy the convolution2D.cu file in that folder. Open terminal and copile the code using nvcc.



nvcc convolution2D.cu -o convolution2D




An execution file will be created and now we can use this file to run the code.



./convolution2D 5 5 2




This will print 

Input:
  12.6   5.9  11.7  12.0  13.7
   3.0   5.0  11.5   4.2   8.3
   7.2   9.4   5.5   7.7  14.3
  13.7   9.5  10.8   2.1   9.1
   0.2   3.6   2.1  12.1   2.4

Kernel:
   6.0   1.9
   1.6  15.0

Output:
 186.7 258.9 197.5 230.0  95.8
 199.5 156.9 215.3 267.8  73.3
 235.7 256.7 120.6 214.0 100.8
 173.8 118.6 267.8  85.4  58.6
  11.9  45.6  39.7  77.1  14.1




While computing throughput we need to compile the code including -G to include debug symbols in the binary.



nvcc -G convolution2D.cu -o convolution2D



And for performace monitoring 


nvprof ./convolution2D 400 400 32