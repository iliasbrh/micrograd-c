# Micrograd for Deep Learning  

First C project : handmade ML library to train on MNIST, using only `stdlib` and `math`. The autograd system is done value-wise with the micrograd style from Andrej Karpathy [here](https://github.com/karpathy/micrograd)  

I reached 94.35% accuracy on the test set, which could be enhanced by adding dropout, by using a learning rate scheduler, or simply by making a deeper model (which is impossible with micrograd given the memory needed).  
Nothing here is optimized for performance, it is just to apply theory and understand things properly.  
Memory management is done with an arena allocator to make allocations and deallocations asymetric thus cleaner.  


# To run on your own  

<pre>
make build
./model.exe
</pre>  


