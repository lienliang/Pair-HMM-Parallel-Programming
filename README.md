# Pair-HMM-Forward-Algorithm for DNA Sequence Alignment

The advanced GPU implementations for Pair-HMM Forward Algorithm with various optimizations, such as efficient host-device communication, task parallelization, pipelining, and memory management strategy, to tackle the challenging task of acclerating the computational-intensive Pair-Hidden Markov Model forward algorithm. The proposed algorithm is implemented with native CUDA C++ and achieved the best speedup of 31:88X on the Pair-HMM workload of a real GATK HaplotypeCaller procedure against the IBM Power8 machine.

Copyright (c) <2021>

<University of Illinois at Urbana-Champaign>

All rights reserved.

Developed by:

<http://dchen.ece.illinois.edu >

<University of Illinois at Urbana-Champaign>
  
This open source project contains the source code for the Pair-HMM Forward Algorithm for DNA Sequence Alignment. 
  
When referencing this application in a publication, please cite the following paper: <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Enliang Li, Subho S. Banerjee, Sitao Huang, Ravishankar K. Iyer, and Deming Chen, “Improved GPU Implementations of the Pair-HMM Forward Algorithm for DNA Sequence Alignment”, *Proceedings of IEEE International Conference on Computer Design*, October 2021.
