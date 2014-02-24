
function e = NEF_MNIST(trainsamples,testsamples,fraction)

training = loadMNISTImages('train-images-idx3-ubyte');



training = num2cell(training,1);
training = training(1:trainsamples);


renormedtrain = cellfun(@(z) arrayfun(@(y) 400*(2*y-1.0),z),training,'UniformOutput',false);


labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels = labels(1:trainsamples);
all_labels = cellfun(@(w) arrayfun(@(z) 400*(2*(z==w)-1.0),labels)',{1,2,3,4,5,6,7,8,9,0},'UniformOutput',false);



layers = cellfun(@(w) create_layer(fraction*28*28,28*28),{1,2,3,4,5,6,7,8,9,0},'UniformOutput',false);

weights = cellfun(@(i) solve_nef(layers{i},renormedtrain,all_labels{i},100),{1,2,3,4,5,6,7,8,9,10},'UniformOutput',false);

rmse_train = mnist_total_error(layers,weights,renormedtrain,labels)

%cellfun(@(i) mnist_error(layers{i},weights{i},renormedtrain,all_labels{i}),{1,2,3,4,5,6,7,8,9,10})

testing = loadMNISTImages('t10k-images-idx3-ubyte');
testing = num2cell(testing,1);
testing = testing(1:testsamples);

renormedtest = cellfun(@(z) arrayfun(@(y)400*(2*y-1.0),z),testing,'UniformOutput',false);

testlabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
testlabels = testlabels(1:testsamples);
all_testlabels = cellfun(@(w) arrayfun(@(z) 400*(2*(z==w)-1.0),testlabels)',{1,2,3,4,5,6,7,8,9,0},'UniformOutput',false);


rmse_test = mnist_total_error(layers,weights,renormedtest,testlabels)
%cellfun(@(i) mnist_error(layers{i},weights{i},renormedtest,all_testlabels{i}),{1,2,3,4,5,6,7,8,9,10})

exit()
