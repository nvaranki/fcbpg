A very simple classic neural network. It's been built on two fully connected layers of 4\*32\*2 schema, amended with bias. Custom backpropagation in use.

The net was trained to determine whether a 4-bit number is less than 8 or not.

Sample output:

* 0000 +1,066 +0,000 
* 0001 +1,015 +0,000 
* 0010 +1,040 +0,000 
* 0011 +0,974 +0,000 
* 0100 +1,030 +0,000 
* 0101 +0,933 +0,000 
* 0110 +1,052 +0,000 
* 0111 +0,986 -0,000 
* 1000 +0,036 +1,000 
* 1001 +0,002 +1,000 
* 1010 +0,011 +1,000 
* 1011 -0,070 +1,000 
* 1100 +0,016 +1,000 
* 1101 -0,032 +1,000 
* 1110 -0,040 +1,000 
* 1111 -0,075 +1,000 

Epoch #143 loss: 0,043
