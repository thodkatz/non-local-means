## Validation

![validation](docs/validation.png)

If you are going to pass `--debug` option then specific steps of the Octave script implementation of the non-local-means filter will be compared with the equivalent C implementation. As it is depicted, we check for 1) Patch creation, 2) Distances, 3) Weights, and finally 4) Filtering. For distances, we are referring to the euclidean distance between patches (D) and for the weights, the final weight value (w = exp(-D^2./sigma)) that will be applied to the noised image.

Due to floating point arithmetic, we have some different values regarding the above. So we assume valid a difference less than 0.0001. 
