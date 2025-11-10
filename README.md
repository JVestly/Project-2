## Project 2 in FYS-STK4155

Project authors: Amund Solum Alsvik, Jørgen Vestly & Kristoffer DH Stalker  

  

---
### Program summary  

This program was created for use in a project studying the performance of back-propagated feed forward neural networks (FFNN). The program contains methods for training models on multiclass classification and one-dimentional regression, and test-files measuring performance for different depths and optimization schemes.    
The regression implementation is coded with the Runge function in mind, but can be easily utilized on other problems by altering or replacing the data formatting function 'create_and_scale_data()' found in 'functions.py'.  
The project can be reached through the side menu, via "FYS-STK4155_Project2".


---  
### Contents   

**Code**  
The Code folder contains:

- classes.py: The NeuralNetwork and GradientDescent classes, which make up the framework of the program.  
- functions.py: contains all utilities used in the project such as cost- and activation functions, scaling, data creation etc.
- imports.py: contains all imports needed for the running the code.
- The following test files in jupyter notebooks, useful for benchmarking and as setup examples:  
  - tests_b.ipynb: contains all tests for task (b), and testing for hyperparameters etc.  
  - tests_b2.ipynb: all tests for (b) but with version 2 of SGD, using only one minibatch per epoch.
  - tests_c.ipynb: contains tests for benchmarks and validation. With and without regularization.  
  - tests_d.ipynb: contains all tests for task (d), with hyperparameter tweaking. Also, it contains a part on overfitting.  
  - tests_d2.ipynb: contains tests for (d), but with the use of SGD_v2, as in tests_b2.ipynb.
  - helper_d.ipynb: used for parallell processing when tests took long.
  - tests_e.ipynb: simply a test-file for testing that L1 and L2 works.  
  - tests_f.ipynb: all tests for task (f), doing classification.  
  - helper_f.ipynb: contains extensive hyperparameter testing for (f).  
Note that in the notebook files they have already been run for project output, but when re-running a particular cell, running the preceding cells may be necessary.  
  
  
**Figures**  
The Figures folder contains all figures used for tests in the project. The plots, maps, and tables all have suitable headlines describing the figure. Figures with the prefix V2 denote figures for the second version of our train_SGD method, using one minibatch per epoch.
This method was often found to perform better for larger datasets (~1e5 datapoints).    
The prefix 'HM' denotes heatmap figures.

All required packages to run the code, are listed in requirements.txt

In the LLM folder, we have added links to our LLM chats (see LLM/links.md).

