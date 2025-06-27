### Quick Start (Windows, macOS, Linux)


This code helps anyone run a local LLM system with RAG allowing a great context use. The LLM will produce results based on the context only, site it's sources as well have reduced hallucination.

The system and the sub-system uses best in class techniques to produce great results, with minimal computational cost whilst maintaining high quality levels. 

The code is designed so the user can replace each part(modular) to personalize for each use case.

Some popular changes could be:
- Use different LLM models, easily plugged in via huggingface library(requiring GGML format).\
- Use unique retrieval systems from LLamaindex, with possible semantic, or hybrid methods.
- Use more types of data formats.

You need to run the command
bash run.sh

This downloads the necessary files,( though each system is different hence it very difficualt to generalize the libraries across the board).

The model should run with taking about a minute or two to produce the results, please be patient as it depends on each machine.
If the model is not using any GPU the results will be very poor and take a lot of time.
