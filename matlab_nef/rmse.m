
function e = rmse(layer,weights,xvals,targets)

  vals = arrayfun(@(z) eval_layer(layer,weights,z),xvals);
num = size(vals);
num = num(1);

e = sqrt(dot(vals-targets,vals-targets)/num);
