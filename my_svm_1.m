clc
types_data = grp2idx(data{:, 1});clear
data = readtable("HepatitisCdata.csv");

% classification 
bt = readtable("blood_val.xlsx");
bt_1 = table2array(bt);
G = abs(10*randn(614,10));
G(:,1:1:10) = bt_1(2:615,:);
y = types_data(2:615);

rand_num = randperm(size(G,1));
G_train = G(rand_num(1:round(0.8*length(rand_num))),:); %we chose 80 random values 
y_train = y(rand_num(1:round(0.8*length(rand_num))),:); %we chose random labeled values

G_test = G(rand_num(round(0.8*length(rand_num))+1:end),:);
y_test = y(rand_num(round(0.8*length(rand_num))+1:end),:);


% Cross Validation partition
c = cvpartition(y_train,'k',5);

% feature selection
opts = statset('display','iter');
classf = @(train_data, train_labels, test_data, test_labels)...
    sum(predict(fitcecoc(train_data, train_labels,"Learners","svm"), test_data) ~= test_labels);

[fs, history] = sequentialfs(classf, G_train, y_train, 'cv', c, 'options', opts,'nfeatures',5);
% this shows sequences works for cross validation
% we are determining the feature number for our model 
% we can change it for the accuracy value

% best hyperparameter
% for the classification
G_train_best_feature = G_train(:,fs);

Mdl_1 = fitcecoc(G_train_best_feature,y_train,"Learners","svm",'OptimizeHyperparameters','auto',...
      'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
      'expected-improvement-plus','ShowPlots',true));
% test
G_test_best_feature = G_test(:,fs);
test_accuracy = sum((predict(Mdl_1,G_test_best_feature) == y_test))/length(y_test)*100;

%plot results
figure;
hgscatter = gscatter(G_train_best_feature(:,1),G_train_best_feature(:,2),y_train);
hold on;
h_sv=plot(Mdl_1.ClassNames(:,1),'ko','markersize',8);

gscatter(G_test_best_feature(:,1),G_test_best_feature(:,2),y_test,'rb','xx');

