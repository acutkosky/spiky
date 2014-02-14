



function a = aval(N,x)

  x = cell2mat(x);
  if(N.alpha*dot(N.e,x)+N.J_bias <= N.J_th)
    a = 0.0;
  else
    a = 1.0/(N.tau_ref-N.tau_RC*log(1.0-N.J_th/(N.alpha*dot(N.e,x)+N.J_bias)));
  end
