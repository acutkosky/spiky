

function e = mnist_total_error(layers,weights,xvals,targets)

  vals = arrayfun(@(z) cellfun(@(i) eval_layer(layers{i},weights{i},z),{1,2,3,4,5,6,7,8,9,10}),xvals,'UniformOutput',false);

vals = cellfun(@(x) mod(find(x==max(x)),10),vals);

num = size(vals);
num = num(2);

correct = arrayfun(@(z) mod(vals(z),10)==mod(targets(z),10),1:num);

e = 1-sum(correct)/num;
