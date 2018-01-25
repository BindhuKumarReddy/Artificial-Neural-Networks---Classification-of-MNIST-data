clc
clear all
close all
load('usps2.mat')
train_data=x1;
train_label=y1;
train_label(train_label==-1)=0;
test_data=train_data;
test_label=train_label;
epochs=3000;
alpha=0.1; % learning rate
beta = 0.01;
feat_size=size(train_data,1); % number of neurons in input layer
n_hidden1=300; % number of neurons in first hidden layer
n_hidden2=200; % number of neurons in second hidden layer
out_neur=size(train_label,1);
output_dim=10; % number of neurons in output layer
% initializing random weights
theta_one=2*rand(feat_size+1,n_hidden1)-1;
theta_two=2*rand(n_hidden1+1,n_hidden2)-1;
theta_three=2*rand(n_hidden2+1,out_neur)-1;
% backpropagation begins
for i=1:epochs
    disp('iteration');disp(num2str(i)); 
    for j=1:length(train_label)
        input=[1;train_data(:,j)];
        hidden_out1=[1;sigmf(theta_one'*input,[beta 0])];
        hidden_out2=[1;sigmf(theta_two'*hidden_out1,[beta 0])];
        output=(sigmf(theta_three'*hidden_out2,[beta 0]))';
        % error at ouput
        err_output=(output'-train_label(:,j)).*(output').*(1-output');
        % error at hidden layers 1 & 2
        err_hidden2=(theta_three*err_output).*(hidden_out2).*(1-hidden_out2);
        err_hidden2=err_hidden2(2:201,1);
        err_hidden1=(theta_two*err_hidden2).*(hidden_out1).*(1-hidden_out1);
        err_hidden1=err_hidden1(2:301,1);
        % updating the weights
        theta_one=theta_one-alpha*(input*err_hidden1');
        theta_two=theta_two-alpha*(hidden_out1*err_hidden2');
        theta_three=theta_three-alpha*(hidden_out2*err_output');
    end
   
end
my_out=zeros(10,4649);

for k=1:length(train_label)
    input=[1;train_data(:,k)];
    hidden_out1=[1;sigmf(theta_one'*input,[beta 0])];
    hidden_out2=[1;sigmf(theta_two'*hidden_out1,[beta 0])];
    output=(sigmf(theta_three'*hidden_out2,[beta 0]))';
    [val,ind]=max(output);
    my_out(ind,k)=1;
end
accur=0;

for i=1:length(y1)
    if find(my_out(:,i)==1)==find(train_label(:,i)==1)
        accur=accur+1;
    else 
        accur=accur+0;
    end
end
disp(accur/4649)

%% y2 prediction
test_data=x2;
% train_label is to be predicted
predict_label=zeros(10,4649);
for k=1:length(train_label)
    input=[1;test_data(:,k)];
    hidden_out1=[1;sigmf(theta_one'*input,[beta 0])];
    hidden_out2=[1;sigmf(theta_two'*hidden_out1,[beta 0])];
    output=(sigmf(theta_three'*hidden_out2,[beta 0]))';
    [val,ind]=max(output);
    predict_label(ind,k)=1;
end