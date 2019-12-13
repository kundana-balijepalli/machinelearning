%loading the data file
load('Diabetes.mat');

diabetes = Diabetes;


[a,b] = bayes(80,diabetes,2,3);
