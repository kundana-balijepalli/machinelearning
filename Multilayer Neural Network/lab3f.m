%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BME777: LAB 3: Multilayer Neural Networks.
% Statlog (Heart) Dataset: https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29
% The first two features are contained in the first two columns.
% 1st feature: Resting blood pressure.
% 2nd feature: Oldpeak = ST depression induced by exercise relative to rest.
% The third column contains the label information.
% Class +1: Absence of heart disease.
% Class -1: Presence of heart disease.
% 50 samples were extracted for each class.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% 1. Eta: Learning rate.
% 2. Theta: Threhold for the cost function to escape the algorithm.
% 3. MaxNoOfIteration: Maximum number of iteration.
% 4. Problem: 1: XOR, otherwise: Classification problem with given dataset.
% 5. Data: the dataset used for training NN when problem ~=1.
% Outputs:
% 1. J: an array of cost.
% 2. w: trained weight matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example of use:
%load ('DataLab3.mat');
%Data = DataLab3;
%Eta = 0.1;
%Theta = 0.001;
%MaxNoOfIteration = 300;
%Problem = 2;
% [J,w] = lab3(Eta,Theta,MaxNoOfIteration,Problem,0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 function [J,w] = lab3f(Eta ,Theta, MaxNoOfIteration, Problem, Data)

%Initialization.

if Problem == 1
    wih1 =[0.69 0.39 0.41]; %weight vector input to hidden unit no.1.
    wih2 =[0.65 0.83 0.37]; %weight vector input to hidden unit no.2.
    who1 =[0.42 0.59 0.56]; %weight vector hidden to output unit. 
    
    % Add data to feature 1,2 and label vectors.
    x1 = [-1 -1 1 1];
    x2 = [-1 1 -1 1];
    t = [-1 1 1 -1];
else
    wih1 =[0.69 0.39 0.41]; %weight vector input to hidden unit no.1.
    wih2 =[0.65 0.83 0.37]; %weight vector input to hidden unit no.2.
    who1 =[0.42 0.59 0.56]; %weight vector hidden to output unit. 
    
    % Add data to feature 1,2 and label vectors.
    x1 = Data(:,1)';
    x2 = Data(:,2)';
    t = Data(:,3)';
end
    
% Initialize number of iteration and cost.
r = 0;
J = zeros(MaxNoOfIteration,1);

while(1);
    
    r=r+1;
    
    % Initialize gradients of the three weight vectors.
    DeltaWih1 = [0 0 0]; % Inputs of bias, x1,x2 to hidden neuron 1.
    DeltaWih2 = [0 0 0]; % Inputs of bias, x1,x2 to hidden neuron 2.
    DeltaWho1 = [0 0 0]; % Inputs of bias, y1,y2 to output neuron.
    
    % Initialize training sample order and predicted output.
    m = 0;
    Z = zeros(1,length(x1));
    
    while(m<length(x1))
        
        m = m + 1;
          
        Xm = [1 x1(1,m) x2(1,m)];
        netj_1 = wih1*Xm';
        netj_2 = wih2*Xm';
        y1 = tanh(netj_1);
        y2 = tanh(netj_2);
        y1s(r)=y1;
        y2s(r)=y2;
        Ym = [1 y1 y2];
        netk_1 = who1*Ym';
        zk = tanh(netk_1);
        Z(m) = zk;
        tk = t(:,m);
                
        % Calculate the sensitivity value of each hidden neuron and the output neuron.
        DeltaO1 = (tk - zk)*(1 - zk^2);% Sensitivity value of the output neuron.
        DeltaH1 = DeltaO1*who1(2)*(1-(y1^2)); % Sensitivity value of hidden neuron 1.
        DeltaH2 = DeltaO1*who1(3)*(1-(y2^2)); % Sensitivity value of hidden neuron 2.
        
        % Update the gradient.
        DeltaWih1 = DeltaWih1 + DeltaH1.*Xm.*Eta;
        DeltaWih2 = DeltaWih2 + DeltaH2.*Xm.*Eta;
        DeltaWho1 = DeltaWho1 + DeltaO1.*Ym.*Eta;           
        
    end
    
    % Update the weight vectors.
    wih1 = wih1 + DeltaWih1; % Weight vector input to hidden unit no.1
    wih2 = wih2 + DeltaWih2; % Weight vector input to hidden unit no.2
    who1 = who1 + DeltaWho1; % Weight vector hidden to output unit.
    
    % Check the condition to stop.
     J(r)= 0.5*(sum((t-Z).^2));

    if(J(r)==Theta)
        break;
    end% y1s(r)=wih1*(y1);
% y2s(r)=y2;

    if(r == MaxNoOfIteration)
        break;
    end
end

% wih1
% 
% wih2
% 
% who1

w = [wih1; wih2; who1];

plot(J);
title('Learning Curve'); 

title('Gradient of Learning Curve'); 
figure;
gscatter(x1, x2, t)
xlabel('x1');
ylabel('x2');
hold on;
x1len=(-length(x1): length(x1));
x2len=(-wih1(:, 1)-wih1(:, 2)*x1len)/wih1(:,3);
plot(x1len, x2len);
x2len=(-wih2(:, 1)-wih2(:, 2)*x1len)/wih2(:,3);
plot(x1len, x2len);
title('Decision Surface for X1-X2');
hold off
figure;
% Xmm=zeros(length(x1),3);
% for i=length(x1)
%     Xmm(i,:) = [1 x1(:,i) x2(:,i)];
% end
% y1s=tanh(wih1*Xmm');
% y2s=tanh(wih2*Xmm');
y1len=(-length(y1):length(y1));
y2len=(-who1(:,1)-who1(:,2)*y1len)/(who1(:,3));
n=1:length(t);
hold on
gscatter(y1s(1,n),y2s(1,n),t);
plot(y1len,y2len);
title('Decision Surface for Y1-Y2');
hold off;

    count=0;
      
      for q = 1:length(Z)
          out =  (Z(1,q))*t(1,q);
          if out > 0    
            count = count + 1;
          end
      end   
      Accuracy = (count/length(x1))*100;
      fprintf('Classification Accuracy: %d \n', Accuracy);

%% Plotting Decision Space
x1m = -1:0.01:1;
x2m = -1:0.01:1;
[X1,X2] = meshgrid(x1m, x2m);

Xall=zeros((length(x1m)*length(x2m)),3);
Xall(:,1)=1;

k=1;
for i = 1:length(x1m)
    for j = 1:length(x2m)
        Xall(k,2) = X1(i,j);
        Xall(k,3) = X2(i,j);
        k=k+1;
    end
end

Zmm=zeros(length(x1m)*length(x2m),1);
Ymm1=zeros(length(x1m)*length(x2m),1);
Ymm2=zeros(length(x1m)*length(x2m),1);

for o=1:(length(x1m)*length(x2m));
    netjs_1=(wih1)*Xall(o,:).';
    netjs_2=(wih2)*Xall(o,:).';
    YY1=(exp(netjs_1)-exp(-netjs_1))/(exp(netjs_1)+exp(-netjs_1));
    YY2=(exp(netjs_2)-exp(-netjs_2))/(exp(netjs_2)+exp(-netjs_2));
    Ymm1(o)=YY1;
    Ymm2(o)=YY2;
    netks=(who1)*[1 YY1 YY2].';
    zks(o)=tanh(netks);
    if zks(o)>0
        Zmm(o)=1;
    else
        Zmm(o)=-1;
    end
    
end

Xmm1=reshape(Xall(:,2),[length(x1m),length(x2m)]);
Xmm2=reshape(Xall(:,3),[length(x1m),length(x2m)]);
Zmm=reshape(Zmm,[length(x1m),length(x2m)]);
Ymm1=reshape(Ymm1,[length(x1m),length(x2m)]);
Ymm2=reshape(Ymm2,[length(x1m),length(x2m)]);


figure;
surf (Xmm1,Xmm2,Zmm);
colormap winter;
shading interp;
xlabel('X1');
ylabel('X2');
title ('X-Space');


figure;
mesh (Ymm1,Ymm2,Zmm);
colormap winter;
shading interp;
xlabel('X1');
ylabel('X2');
title ('Y-Space');
      
 end 


