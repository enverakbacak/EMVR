
clear all;
clc;

x=linspace(0,350,1);


load('Evaluations/NR_EMIR_Barcelona_16@3_5d.mat');
plot(n_DCG_mean, '-k', 'LineWidth',4); 
hold on

load('Evaluations/NR_EMIR_Barcelona_32@3_5d.mat');
plot(n_DCG_mean,'-g', 'LineWidth',3); 
hold on

load('Evaluations/NR_EMIR_Barcelona_64@3_5d.mat');
plot(n_DCG_mean,'-b',  'LineWidth',3);
hold on

%load('Evaluations/R_EMIR_Barcelona_16@3_2d.mat');
%plot(n_DCG_mean,'-r',  'LineWidth',4');
%hold on
%}

%{
load('Evaluations/R_EMIR_Barcelona_16@3_3d.mat');
plot(n_DCG_mean, '-k', 'LineWidth',4); 
hold on

load('Evaluations/R_EMIR_Barcelona_32@3_3d.mat');
plot(n_DCG_mean,'-g', 'LineWidth',3); 
hold on

load('Evaluations/R_EMIR_Barcelona_64@3_3d.mat');
plot(n_DCG_mean,'-b',  'LineWidth',3);
hold on

%load('Evaluations/R_EMIR_Barcelona_16@3_2d.mat');
%plot(n_DCG_mean,'-r',  'LineWidth',4');
%hold on
%}



%{
load('Evaluations/R_EMIR_Barcelona_16@3_2d.mat');
plot(n_DCG_mean, '--k', 'LineWidth',3); 
hold on

load('Evaluations/R_EMIR_Barcelona_16@3_3d.mat');
plot(n_DCG_mean,'--g', 'LineWidth',3); 
hold on

load('Evaluations/R_EMIR_Barcelona_16@3_4d.mat');
plot(n_DCG_mean,'-b',  'LineWidth',3);
hold on

load('Evaluations/');
plot(n_DCG_mean,'-g',  'MarkerSize',6, 'MarkerFaceColor',[0 1 0],  'LineWidth',3');
hold on

load('Evaluations/');
plot(n_DCG_mean,'-r',  'LineWidth',3);
hold on

load('Evaluations/');
plot(n_DCG_mean,'-k',  'LineWidth',3');
hold on
%}

%{
load('Evaluation_Revision/streetsDataset/DMQR/streets_5d_256_DMQR_n_DCG_mean');
plot(n_DCG_mean,'-r',   'LineWidth',4');
%plot(n_DCG_mean,'-rd',  'MarkerSize',12, 'MarkerFaceColor',[1 0 0],  'LineWidth',1');
hold on

load('Evaluation_Revision/streetsDataset/DMQR/streets_5d_512_DMQR_n_DCG_mean');
plot(n_DCG_mean,'-k',   'LineWidth',4');
hold on
%}

%{
load('Evaluation_Revision/lamdaDataset/regParam/5d_128_DMQR_n_DCG_mean');
plot(n_DCG_mean,'-gd',  'MarkerSize',10, 'MarkerFaceColor',[0 1 0], 'LineWidth',1');
hold on

load('Evaluation_Revision/lamdaDataset/regParam/5d_256_DMQR_n_DCG_mean');
plot(n_DCG_mean,'-bo', 'MarkerSize',10, 'MarkerFaceColor',[0 0 1], 'LineWidth',1);
hold on

load('Evaluation_Revision/lamdaDataset/regParam/2d_512_001_DMQR_n_DCG_mean');
plot(n_DCG_mean,'-rs',  'MarkerSize',10, 'MarkerFaceColor',[1 0 0],  'LineWidth',1');
hold on
%}


set(gca,'FontSize',46);
%title('EMR Method for Lamda = 0.1' ,'FontSize', 30)

%title('16-bits hash code' ,'FontSize', 30)

ylabel('nDCG scores.' ,'FontSize', 36)
xlabel('The Number of Retrieved Images' ,'FontSize', 36) 

%legend({'2-queries', '3-queries','4-queries', '5-queries', },'Location','southwest' ,'FontSize', 38)
legend({'16-bit', '32-bit', '64-bit', },'Location','southwest' ,'FontSize', 38)
%legend({'Lamda = 0.01', 'Lamda = 0.1', 'Lamda = 0.7', 'Lamda = 1', },'Location','southwest' ,'FontSize', 38)








