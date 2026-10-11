# Micrograd for Deep Learning  

First C project : handmade ML library to train on MNIST, using only `stdlib` and `math`. The autograd system is done value-wise with the autograd style from Andrej Karpathy [here](https://github.com/karpathy/micrograd)  

I reached 90.53% accuracy on the test set, which could be enhanced by implementing Adam optimizer, by adding dropout, by using a learning rate scheduler, or simply by making a deeper model.  
Nothing here is optimized for performance, it is just to apply theory and understand things properly.  
Memory management is done with an arena allocator to make allocations and deallocations asymetric thus cleaner.  


# To run on your own  

<pre>  
gcc arena.c value.c matrix.c read_ubyte.c main.c -o model.exe  
./model.exe  

</pre>  
  
or replace gcc with the compiler you use.


