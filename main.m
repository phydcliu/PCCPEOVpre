clc
clear all
S_Data=xlsread();
n = 2;
m = 1
train_num = 1500;
train_Data = S_Data(1:train_num,:);
[train_Input,minI,maxI] = premnmx(train_Data(:,1:n)');%train_Data(:,1:n)'//premnmx
[train_Output,minO,maxO] = premnmx(train_Data(:,n+1:end)');%premnmx
test_Data = S_Data(train_num+1:end,:);
test_Input = tramnmx(test_Data(:,1:n)',minI,maxI);
test_Output = tramnmx(test_Data(:,n+1:end)',minO,maxO);
RMSE = []; 
Gam = 1:10:200;
sig = 600; 
for q = 1:20     
    gam = Gam(1,q);
    tic; 
    [alpha, b] = trainlssvm({train_Input',train_Output','f',gam,sig});
   [SVMtest_Output, Zt] = simlssvm({train_Input',train_Output','f',gam,sig}, test_Input');
    toc;
    test_Output = postmnmx(test_Output,minO,maxO);
    SVMtest_Output = postmnmx(SVMtest_Output,minO,maxO);
    test_err = test_Output' - SVMtest_Output;
    n1 = length(SVMtest_Output);
    test_RMSE = sqrt(sum((test_err).^2)/n1);
    RMSE(1,q) = test_RMSE;
end

train_Output = postmnmx(train_Output',minO,maxO);
SVMtrain_Output = postmnmx(SVMtrain_Output',minO,maxO);
train_err = train_Output - SVMtrain_Output';
n1 = length(SVMtrain_Output);
train_RMSE = sqrt(sum((train_err).^2)/n1);
test_Data = S_Data(train_num+1:end,:);
test_Input = tramnmx(test_Data(:,1:n)',minI,maxI)';
test_Output = tramnmx(test_Data(:,n+1:end)',minO,maxO)';
SVMtest_Output = simlssvm({train_Input',train_Output,type,gam,sig,'RBF_kernel','preprocess'},{alpha,b},test_Input);
test_Output = postmnmx(test_Output,minO,maxO);
SVMtest_Output = postmnmx(SVMtest_Output',minO,maxO);
test_err = test_Output - SVMtest_Output';
n2 = length(SVMtest_Output');
test_RMSE = sqrt(sum((test_err).^2)/n2);
figure(6); 
subplot(2,1,1); 
plot(SVMtest_Output,':og'); 
hold on;
plot(test_Output','-*b');  
