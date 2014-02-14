
function l = create_layer(s,d)

  nA = 0.000000001;
  ms = 0.001;

  e = cellfun(@(x) x/sqrt(dot(x,x)),num2cell(randn(d,s),1),'UniformOutput',false);

  alpha = num2cell(0.1/400 * (17*nA + 5*nA*randn(1,s)),1);
 

  J_bias = num2cell(7*nA + 20*nA*(2*rand(1,s)-1.0));


  tau_ref = num2cell(1.5*ms +0.3*ms*randn(1,s));


  tau_RC = num2cell(20*ms + 4*ms*randn(1,s));

  J_th = num2cell(1*nA +0.2*nA*randn(1,s));


  l = struct('e',e,'alpha',alpha,'J_bias',J_bias,'tau_ref',tau_ref,'tau_RC',tau_RC,'J_th',J_th);

