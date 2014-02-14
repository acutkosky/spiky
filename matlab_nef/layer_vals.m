
function a = layer_vals(layer,x)
  a = arrayfun(@(z) aval(z,x),layer);
