% Example of use:
% load DataLab4.mat;
% Data = Breast_Tissue;
% InitMean1 = [choose your mean1 ];
% InitMean2 = [choose your mean2 ];
% InitMean3 = [choose your mean3 ];
% MaxNoOfIteration = 400;
% [FinMean1, FinMean2, FinMean3,Label] = lab4(Data,InitMean1,InitMean2,InitMean3,MaxNoOfIteration);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load DataLab4.mat;
Data = Breast_Tissue;
InitMean1 = [420 0.45 670];
InitMean2 = [180 2.3 750];
InitMean3 = [260 0.87 1111];
%InitMean3 = 0;
MaxNoOfIteration = 400;

%% Initialization
if InitMean3~=0
   PrevMean = [InitMean1; InitMean2; InitMean3];
else
   PrevMean = [InitMean1; InitMean2];
end

Label = zeros(length(Data),1);
Itr = 0;

%% Calculating means
while(1)
    Itr = Itr + 1;
    %%%%%%%%%%%%Compute Euclidean distance from each sample to the given means%%%%%%%%%%%%
    for i=1:length(Data)
       D1 = (Data(i,1) - PrevMean(1,1))^2 + (Data(i,2) - PrevMean(1,2))^2 + (Data(i,3) - PrevMean(1,3))^2;
       D2 = (Data(i,1) - PrevMean(2,1))^2 + (Data(i,2) - PrevMean(2,2))^2 + (Data(i,3) - PrevMean(2,3))^2;
       if InitMean3~=0
           D3 = (Data(i,1) - PrevMean(3,1))^2 + (Data(i,2) - PrevMean(3,2))^2 + (Data(i,3) - PrevMean(3,3))^2;
       end
    %%%%%%%%%%%%%Identify the minimum distance from the sample to the means%%%%%%%%%%%%%%% 
       if InitMean3~=0
           [~,Index] = min([D1 D2 D3]);
       else
           [~,Index] = min([D1 D2]);
       end
    %%%%%%%%%%%%Label the data samples based on the minimum euclidean distance%%%%%%%%%%%%  
       Label(i) = Index;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%Compute the new means%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    FinalMean1 = mean( Data(Label==1,:) );
    FinalMean2 = mean( Data(Label==2,:) );
    if InitMean3~=0
        FinalMean3 =  mean(Data(Label==3,:) );
    else
        FinalMean3 = 0;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%Check for criterion function%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%If criteria not met repeate the above%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if InitMean3~=0
        CurrMean = [FinalMean1; FinalMean2;  FinalMean3];
    else
        CurrMean = [FinalMean1; FinalMean2];
    end
    if (CurrMean == PrevMean) % Check conditions to stop the algorithm.
        break;
    end
    if (Itr == MaxNoOfIteration)
        break;
    end
    PrevMean = CurrMean;   
    %% Graphing ********************************************
%Graphing Initial means
   if (Itr == 1)
        x1 = Data(Label==1,1);
        x2 = Data(Label==1,2);
        figure; 
        if InitMean3~=0
            x3 = Data(Label==1,3);
            scatter3(x1,x2,x3,'ko');
            hold on;
            scatter3(CurrMean(1,1),CurrMean(1,2),CurrMean(1,3),'kx');
        else
            plot(x1,x2,'ko');
            plot(CurrMean(1,1),CurrMean(1,2),'kx');
        end

 %For label 2: 
 
        x1 = Data(Label==2,1);
        x2 = Data(Label==2,2);
        if InitMean3~=0
            x3 = Data(Label==2,3);
            scatter3(x1,x2,x3,'mo');
            scatter3(CurrMean(2,1),CurrMean(2,2),CurrMean(2,3),'mx');
        else
            plot(x1,x2,'mo');
            plot(CurrMean(2,1),CurrMean(2,2),'mx');
            legend('Label-1','Mean-1','Label-2','Mean-2');
            xlabel('x1');
            ylabel('x2');
        end
 %For label 3       
        if InitMean3~=0
            x1 = Data(Label==3,1);
            x2 = Data(Label==3,2);
            x3 = Data(Label==3,3);
            scatter3(x1,x2,x3,'bo');
            scatter3(CurrMean(3,1),CurrMean(3,2),CurrMean(3,3),'bx');
            xlabel('x1');
            ylabel('x2');
            zlabel('x3');
%         else
%             plot(x1,x2,'bo');
%             plot(CurrMean(3,1),CurrMean(3,2),'bx');
%             xlabel('x1');
%             ylabel('x2');
        legend('Label-1','Mean-1','Label-2','Mean-2','Label-3','Mean-3');
        end
        title('Initial K Means Classification');
%         legend('Label-1','Mean-1','Label-2','Mean-2','Label-3','Mean-3');
        hold off;
   end
end % while loop

%% Final Iteration graph 
x1 = Data(Label==1,1);
x2 = Data(Label==1,2);
figure; 
if InitMean3~=0
    x3 = Data(Label==1,3);
    scatter3(x1,x2,x3,'ko'); hold on;
    scatter3(CurrMean(1,1),CurrMean(1,2),CurrMean(1,3),'kx');
else
    plot(x1,x2,'ko');
    plot(CurrMean(1,1),CurrMean(1,2),'kx');
end

x1 = Data(Label==2,1);
x2 = Data(Label==2,2);
if InitMean3~=0
    x3 = Data(Label==2,3);
    scatter3(x1,x2,x3,'mo');
    scatter3(CurrMean(2,1),CurrMean(2,2),CurrMean(2,3),'mx');
else
    plot(x1,x2,'mo');
    plot(CurrMean(2,1),CurrMean(2,2),'mx');
end

if InitMean3~=0
    x1 = Data(Label==3,1);
    x2 = Data(Label==3,2);
    x3 = Data(Label==3,3);
    scatter3(x1,x2,x3,'bo');
    scatter3(CurrMean(3,1),CurrMean(3,2),CurrMean(3,3),'bx');
     xlabel('x1');
     ylabel('x2');
     zlabel('x3');
% else
%     plot(x1,x2,'bo');
%     plot(CurrMean(3,1),CurrMean(3,2),'bx');
%     xlabel('x1');
%     ylabel('x2');
end
title('Final K Means Classification');
legend('Label-1','Mean-1','Label-2','Mean-2','Label-3','Mean-3')
hold off;

FinalMean1 = FinalMean1';
FinalMean2 = FinalMean2';
FinalMean3 = FinalMean3';
disp('Final means of the clusters: ');
table(FinalMean1,FinalMean2,FinalMean3)
