Project 2 in FYS-STK4155

Project authors: Amund Solum Alsvik, Jørgen Vestly & Kristoffer DH Stalker


In this project, we investigate the use of neural networks for regression and classification. We trained on different network depths, and compared the results to regression and classification without the use of neural networks. For the report, see "FYS-STK4155_Project2" in the side menu. 

The Code folder is organised the following:

classes.py: contains the NeuralNetwork and GradientDescent class needed for this project.
functions.py: contains all utilities used in the project such as cost- and activation functions, scaling, data creation etc.
imports.py: contains all imports needed for the running the code.

To run the code, choose a particular task and go to tests_task.ipynb, and run all:

tests_b.ipynb: contains all tests for task (b), and testing for hyperparameters etc.  
tests_b2.ipynb: all tests for (b) but with version 2 of SGD, using only one minibatch per epoch.
tests_c.ipynb: contains tests for benchmarks and validation. With and without regularization.  
tests_d.ipynb: contains all tests for task (d), with hyperparameter tweaking. Also, it contains a part on overfitting.  
tests_d2.ipynb: contains tests for (d), but with the use of SGD_v2, as in tests_b2.ipynb.
helper_d.ipynb: used for parallell processing when tests took long.
tests_e.ipynb: simply a test-file for testing that L1 and L2 works.  
tests_f.ipynb: all tests for task (f), doing classification.  
helper_f.ipynb: contains extensive hyperparameter testing for (f).


Note that the commited notebooks are already runned, but to replicate the same results or to change parameters, run all.

All required packages to run the code, are listed in requirements.txt

In the LLM folder, we have added links to our LLM chats (see LLM/links.md).

In the folder, Figures, we have added all figures used for tests in this project. The plots, maps, and tables all have suitable headlines, describing the figure. Figures with a prefix, V2 contains all figures for the second version of our train_SGD method, using one minibatch per epoch.
The prefix, HM, are heatmap figures.
