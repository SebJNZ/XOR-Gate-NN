# Compile and Run
1. Compile both scripts:
```
g++ main.cpp -o main
g++ -DNETWORKTESTING nn_test.cpp main.cpp -o nn_test
```

2. Train the network:
```
$ ./main
Epoch 0 | Wrongness: 0.0647719 | Output: 0.359922
Epoch 0 | Wrongness: 0.249033 | Output: 0.294262
Epoch 0 | Wrongness: 0.234507 | Output: 0.315154
Epoch 0 | Wrongness: 0.0348447 | Output: 0.263987
Epoch 200 | Wrongness: 0.148547 | Output: 0.545063
Epoch 200 | Wrongness: 0.131323 | Output: 0.48751
Epoch 200 | Wrongness: 0.116816 | Output: 0.516645
Epoch 200 | Wrongness: 0.106879 | Output: 0.462339
Epoch 400 | Wrongness: 0.142361 | Output: 0.533594
Epoch 400 | Wrongness: 0.132558 | Output: 0.485106
Epoch 400 | Wrongness: 0.11439 | Output: 0.52169
Epoch 400 | Wrongness: 0.10939 | Output: 0.467739
.................................................
Epoch 99400 | Wrongness: 5.68022e-05 | Output: 0.0106585
Epoch 99400 | Wrongness: 7.08076e-05 | Output: 0.9881
Epoch 99400 | Wrongness: 5.71492e-05 | Output: 0.989309
Epoch 99400 | Wrongness: 4.46305e-05 | Output: 0.0094478
Epoch 99600 | Wrongness: 5.66716e-05 | Output: 0.0106463
Epoch 99600 | Wrongness: 7.06448e-05 | Output: 0.988113
Epoch 99600 | Wrongness: 5.70231e-05 | Output: 0.989321
Epoch 99600 | Wrongness: 4.45314e-05 | Output: 0.00943731
Epoch 99800 | Wrongness: 5.65416e-05 | Output: 0.0106341
Epoch 99800 | Wrongness: 7.04828e-05 | Output: 0.988127
Epoch 99800 | Wrongness: 5.68975e-05 | Output: 0.989333
Epoch 99800 | Wrongness: 4.44327e-05 | Output: 0.00942685

Final Output after training for input pair - 0 0: 0.0106219

Final Output after training for input pair - 0 1: 0.988141

Final Output after training for input pair - 1 0: 0.989344

Final Output after training for input pair - 1 1: 0.00941614
```

3. Test the network
```
$ ./nn_test
Loaded

Layer: Input
Neurons: 2
Neuron Weights: 0.680375 | Bias: -0.211234 | Output: 0

Neuron Weights: 0.566198 | Bias: 0.59688 | Output: 0


Layer: Hidden 1
Neurons: 4
Neuron Weights: 5.48047 -5.38642 | Bias: 2.75335 | Output: 0

Neuron Weights: 1.4326 -0.216155 | Bias: -1.53717 | Output: 0

Neuron Weights: 3.77377 -3.56815 | Bias: 1.8152 | Output: 0

Neuron Weights: 6.43326 -6.42327 | Bias: -3.43521 | Output: 0


Layer: Output
Neurons: 1
Neuron Weights: -7.57793 -0.000221972 -3.76256 10.887 | Bias: 5.48586 | Output: 0


Enter two binary numbers: 1 0 
0.989345


Enter two binary numbers: 0 0
0.010622


Enter two binary numbers: 1 1
0.00941619


Enter two binary numbers: 0 1
0.988141
```

## Credits
For full disclosure, AI was used to assist me during this project, mostly giving me vague instructions, and the mathematical formulas for the more complex stuff (back-propagation, and forward pass). It was never used to generate code.
