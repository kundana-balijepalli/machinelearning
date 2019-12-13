load('DataLab3.mat')
Data = DataLab3;
Eta = 0.1;
Theta = 0.001;
MaxNoOfIteration = 300;
Problem = 1;

[J,w] = lab3f(Eta,Theta,MaxNoOfIteration,Problem,Data);
