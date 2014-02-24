

function e = mnist_error(layer,weights,xvals,targets)

  vals = arrayfun(@(z) eval_layer(layer,weights,z),xvals);
num = size(vals);
num = num(2);

error = arrayfun(@(z) vals(z)*targets(z)< 0,1:num);

e = sum(error)/num;
