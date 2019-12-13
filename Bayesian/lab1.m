%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BME777: LAB 1: Bayesian Decision Theory.

% Acknowledgement: We thankfully acknowledge UCI Machine Learning Repository for the 
%dataset used in this lab exercise.The data for this lab is extracted from Pima Indians 
%Diabetes Data Set: https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes


% The first two columns contain the 2nd and 3rd features of the original dataset. 
% 1st feature: Plasma glucose concentration.
% 2nd feature: Diastolic blood pressure (mm Hg).
% The third colum contatins the labels (1: positive, 2: negative) for diabetes.
% 268 samples were extracted for each class.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% 1. FeatureX: Feature value to be tested (to identify which class it belongs to).
% 2. Data: Matrix containing the training feature samples and class labels.
% 3. FeatureForClassification: Select type of feature used for
% classification. (1 or 2)
% 4. LabelColumn: Specify the column containing the labels of the data.
% Outputs:
% 1. PosteriorProbabilities: Posterior probabilities of class 1 and 2 given FeatureX.
% 2. DiscriminantFunctionValue: Value of the discriminant function given FeatureX.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example of use:
 load Diabetes.mat;
 FeatureX = 5;
 LabelColumn = 3;
 FeatureForClassification = 1;
% [PosProb, G] = lab1(FeatureX, Diabetes,FeatureForClassification, LabelColumn);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [PosteriorProbabilities,DiscriminantFunctionValue]= bayes(FeatureX,Data,FeatureForClassification, LabelColumn)

% Get number of samples.
[ro,~] = size(Data);

% Separating classes classification (1 or 2).  
class1_index = find(Data(:,3)==1);
class2_index = find(Data(:,3)==2);

class1=Data(class1_index, FeatureForClassification);
class2=Data(class2_index, FeatureForClassification);

% Select feature for classification (1 or 2).  
% SelectedFeature=Data(:,FeatureForClassification);

% Get class labels.
Label=Data(:,3); 

%%%%%%%%Plot the histogram and box plot of features related to two classes%%%%%%%%%%
    
% Plot hist.
figure
histogram(class1);
title('Histogram plot distribuion for each class'); 
hold on
histogram(class2);
legend('Class 1 (postive)','Class 2 - (negative)');
xlabel('Value of "x"');
ylabel('Count');
hold off

% Plot boxplot.
figure
subplot(2,1,1)
boxplot(class1);
title('Box plot distribution of class 1');
hold on
subplot(2,1,2) 
boxplot(class2);
title('Box plot distribution of class 2');
hold off
    
%%%%%%%%%%%%%%%%%%%%%%%Compute Prior Probabilities%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate prior probability of class 1 and class 2.
Pr1= length(class1)./ro; 
Pr2= length(class2)./ro; 

%%%%%%%%%%%%%%%%%%%%Compute Class-conditional probabilities%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate the mean and the standard deviation of the class conditional density p(x/w1).
m11= mean(class1);  
std11= std(class1); 

% Calculate the mean and the standard deviation of the class conditional density p(x/w2).
m12= mean(class2); 
std12= std(class2); 


% %% Threshold 1
% 
% x = 1; %feature value that is plugged into while loop to find optimal threshold
% while true
% % Calculate the class-conditional probability of class 1 and class 2.
%  cp11= (1./(sqrt(2*pi).*(std11))).*exp((-1/2)*((x - m11)/(std11)).^2); 
%  cp12= (1./(sqrt(2*pi).*(std11))).*exp((-1/2)*((x - m12)/(std12)).^2); 
%  evidence = cp11.*Pr1 + cp12.*Pr2;
% % Calculate the posterior probability of class 1 and class 2.
%  pos11= (cp11.*Pr1)/evidence;
%  pos12= (cp12.*Pr2)/evidence;
%  PosteriorProbabilities = [pos11,pos12];
% 
%  % Leave loop when Posterior1 = Posterior2 disp('Discriminant function value for the test feature');
%  Discriminant = pos11 - pos12
%         if (FeatureForClassification == 1)
%             if ((Discriminant) >= 0)
%                 Th1 = x;
%                 break;
%             end
%         end
%         if (FeatureForClassification == 2)
%             if ((Discriminant) <= 0)
%                 Th1 = x;
%                 break;
%             end
%         end
%         x = x + 1;
% end
% 
% %% Threshold 2
% y = 1;
%     while true
%         % Calculate the class-conditional probability of class 1 and class 2.
%         RCCP1 = (1./(sqrt(2*pi).*(std11))).*exp((-1/2)*((y-m11)/(std11)).^2);
%         RCCP2 = (1./(sqrt(2*pi).*(std12))).*exp((-1/2)*((y-m12)/(std12)).^2);
%         evidence2 = RCCP1.*Pr1 + RCCP2.*Pr2;
%         % Calculate the posterior probability of class 1 and class 2.
%         RPosterior1 = ((RCCP1).*(Pr1))./evidence2; 
%         RPosterior2 = ((RCCP2).*(Pr2))./evidence2;
%         RPos1 = 2 * RPosterior2;
%         RPos2 = 10 * RPosterior1;
%         RDiscriminant  = RPos2 - RPos1;
%         if ((RDiscriminant) >= 0)
%                 Th2 = y;
%                 break;
%         end
%         y = y + 1;
%     end

%% Calculations for Posterior Probabilities and Discriminant Function

% Calculate the class-conditional probability of class 1 and class 2.
cp11= (1./(sqrt(2*pi).*(std11))).*exp((-1/2)*((FeatureX - m11)/(std11)).^2); 
cp12= (1./(sqrt(2*pi).*(std11))).*exp((-1/2)*((FeatureX - m12)/(std12)).^2);  

evidence = cp11.*Pr1 + cp12.*Pr2;
%%%%%%%%%%%%%%%%%%%%%%%Compute the posterior probabilities%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Posterior probabilities for the test feature');

% Calculate the posterior probability of class 1 and class 2.
pos11= (cp11.*Pr1)/evidence;
pos12= (cp12.*Pr2)/evidence;

PosteriorProbabilities = [pos11,pos12] ;

%%%%%%%%%%%Compute Discriminant function Value for Min Error Rate Classifier%%%%%%%%
disp('Discriminant function value for the test feature');

% Compute the g(x) for min err rate class.
DiscriminantFunctionValue = pos11 - pos12;

% if (FeatureX > Th1) || (FeatureX > Th2)
%    ClassLabel = 1;
% end
% if (FeatureX < Th1) || (FeatureX < Th2)
%    ClassLabel = 2;
% end

%Printing the results
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
line1 = sprintf('The posterior probabilities are %f and %f',pos11,pos12);
disp(line1);
line2 = sprintf('For this feature, Threshold 1 is %f and threshold 2 is %f',Th1,Th2);
disp(line2);
line3 = sprintf('Discriminant Function g(x) value  is %f',DiscriminantFunctionValue);
disp(line3);
line4 = sprintf('Therefore the class label for the test value is %d',ClassLabel);
disp(line4);

%end

