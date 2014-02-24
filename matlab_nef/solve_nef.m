
function weights = solve_nef(layer,xvals,targets,regularization)


  num = size(xvals);
num = num(1);

A = cell2mat(arrayfun(@(z) layer_vals(layer,z)',xvals,'UniformOutput',false))';


		      dim = size(A);
dim = dim(2);



Lambda = diag(regularization*ones(1,dim));

		      weights = (A'*A+Lambda'*Lambda)^(-1)* A' *(targets');



