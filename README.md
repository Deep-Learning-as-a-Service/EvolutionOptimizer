Evolutionary Algorithm, that can optimize the choice of hyperparameters and architecture of machine learning models.

- in folder "experiments" are all executable files
- use runner.py from the root directory to run the experiments
- define a config.py with the global vars CHAT_ID and BOT_KEY, if you want to use send_telegram


Notes:
- a small unnoticed difference in the instance is enough for the difference, the cross instances, of the very similar instances are again very same!
- Solution:
	- only max one cross instance
	- for crossing minimum dissimilarity, not just stupid the best ones 
	- Metric for similarity
		- n_duplicates in generation
		- n_duplicates only one difference

    
    - crossing of two individuals, that are only different in the same position, the duplicate protection loop will go forever!
    - even if they are not the same, they must differ in 2 attributes


Ideas: 
- if there is no better individual for many generations, the generation could get more random -> adaptive search: instruments provided by DNA! metric, when the new generation was good enough, in crossing, similar values will appear most of the time
- diversity good over attributes?
- How to evolution.
    - you need a very precise fitness function!!! KFold required!
    - the derivation between Folds should be low
    - most importantly the differnce between KFold mean and the final training should be low -> representative
    70 (derivation 1 percent) mean to 72
    69 (over 10 percent derivation) mean to 65
    - write a function the does the check and has as output a good metric that you can see, if an optimization of your fitness function will lead to a better model!!!!
    - cross could be an individual in between
    - no better individual for some generations, the change rate needs to be both bigger and smaller - smaller for micro optimization, bigger for finding a complete new optimum
    - finding another optimum, I could open a  new familiy tree, where the maximum is not compare with my main generation