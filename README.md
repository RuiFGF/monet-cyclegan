(taken from Amy Jang's work at https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial)

# CycleGAN

A Cycle-consistent Adversarial Networks attempts to map two domains by ensuring that the reciprocal mapping leads to the same result. This means that given a source and target domains (respectively, A and B) we are looking for mappings

![equation](https://latex.codecogs.com/svg.image?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20F:%20A%20%5Crightarrow%20B%5C%5C%20G:%20B%20%5Crightarrow%20A%20%5Cend%7Bmatrix%7D%5Cright.%20)

such that

![equation](https://latex.codecogs.com/svg.image?F(G(X))%20%5Capprox%20X%20)


# This model

The goal of this model is to take a photo (domain A) and "translate" it into an artistic style (domain B). We train this model with unpaired data, which means it is not comprised of "direct translations" between a given photo and the same with the expected artistic style. 
